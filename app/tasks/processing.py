"""
Background task that processes an uploaded PDF.

This runs *after* the upload endpoint has already responded to the client.
It picks up where /upload left off: takes the PDF on disk, runs PageIndex
on it, stores the resulting tree, and updates the job's status in SQLite
as it goes.
"""
import asyncio
import json
import traceback
from pathlib import Path

import aiosqlite
import filelock

from app.config import settings
from app.database import DB_PATH
from app.services.pageindex_service import build_tree, PageIndexError


async def update_job(
    job_id: str,
    status: str,
    progress: int,
    error: str | None = None,
) -> None:
    """Update a job row in the jobs table."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE jobs
            SET status = ?, progress = ?, error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
            """,
            (status, progress, error, job_id),
        )
        await db.commit()


async def update_doc_after_processing(
    doc_id: str,
    tree_path: Path,
    page_count: int,
) -> None:
    """Update the documents table once a tree has been built."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE documents
            SET tree_path = ?, page_count = ?
            WHERE doc_id = ?
            """,
            (str(tree_path), page_count, doc_id),
        )
        await db.commit()


def count_pages_in_tree(tree: dict) -> int:
    """Walk the tree to find the highest end_index — that's the page count."""
    max_page = 0

    def walk(nodes: list[dict]) -> None:
        nonlocal max_page
        for node in nodes:
            if "end_index" in node:
                max_page = max(max_page, node["end_index"])
            if node.get("nodes"):
                walk(node["nodes"])

    walk(tree.get("structure", []))
    return max_page


async def process_document(
    job_id: str,
    doc_id: str,
    pdf_path: Path,
    original_filename: str = "",
) -> None:
    """
    The actual background job. Runs after the upload endpoint returns.

    Flow:
      1. Mark job as 'processing' (5%)
      2. Acquire a per-document lock so two concurrent uploads of the
         same PDF don't both run PageIndex
      3. If another process already built the tree while we waited, exit fast
      4. Run PageIndex via the wrapper (this is the expensive step, 5% -> 90%)
      5. Update the documents table with tree_path and page_count
      6. Mark job as 'complete' (100%)

    On any failure, mark the job 'failed' with the traceback in the error column.
    """
    lock_path = settings.data_dir / f"{doc_id}.lock"
    tree_path = settings.trees_dir / f"{doc_id}.json"
    lock = filelock.FileLock(str(lock_path), timeout=600)

    try:
        await update_job(job_id, "processing", 5)

        # Try to acquire the lock. If another process is already building the
        # same doc, we wait for them to finish, then check if their result
        # is usable instead of re-running PageIndex ourselves.
        try:
            lock.acquire(timeout=1)
            we_hold_lock = True
        except filelock.Timeout:
            we_hold_lock = False

        if not we_hold_lock:
            # Someone else is processing this doc. Wait until they're done.
            await update_job(job_id, "processing", 10)
            try:
                # This blocks until the other process releases the lock.
                lock.acquire(timeout=600)
            except filelock.Timeout:
                raise PageIndexError(
                    "Timed out waiting for another worker to finish processing this document"
                )
            we_hold_lock = True

            # Check whether they actually produced a tree
            if tree_path.exists():
                # Great — reuse their work
                tree = json.loads(tree_path.read_text(encoding="utf-8"))
                page_count = count_pages_in_tree(tree)
                await update_doc_after_processing(doc_id, tree_path, page_count)
                await update_job(job_id, "complete", 100)
                return
            # Otherwise fall through — the other worker failed and we'll retry.

        await update_job(job_id, "processing", 15)

        # Run PageIndex. This is the slow part (~1-3 minutes on a 150-page PDF).
        # We run it in a thread executor so it doesn't block FastAPI's event loop.
        loop = asyncio.get_event_loop()
        tree = await loop.run_in_executor(
            None,  # default thread pool
            lambda: build_tree(
                pdf_path=pdf_path,
                doc_id=doc_id,
                original_filename=original_filename,
            ),
        )

        await update_job(job_id, "processing", 95)

        page_count = count_pages_in_tree(tree)
        await update_doc_after_processing(doc_id, tree_path, page_count)
        await update_job(job_id, "complete", 100)

    except Exception as e:
        error_text = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        await update_job(job_id, "failed", 0, error=error_text)

    finally:
        # Always release the lock if we hold it.
        try:
            if lock.is_locked:
                lock.release()
        except Exception:
            pass