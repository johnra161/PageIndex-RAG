import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import aiosqlite

from app.config import settings
from app.database import init_db, get_db, DB_PATH
from app.models import UploadResponse, JobStatus, QueryRequest
from app.utils.hashing import sha256_bytes
from app.tasks.processing import process_document
from app.services.navigation import traverse_tree, synthesize_answer, long_context_answer
from app.utils.cache import get_component, set_component
import time
import json

app = FastAPI(title="Visualized RAG", version="0.1.0")

@app.on_event("startup")
async def startup():
    await init_db()

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    # 2. Read into memory to hash it (validates size too)
    max_bytes = settings.max_pdf_size_mb * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(413, f"File too large. Max {settings.max_pdf_size_mb}MB")

    # 3. Hash = document identity
    doc_id = sha256_bytes(contents)
    pdf_path = settings.pdfs_dir / f"{doc_id}.pdf"
    tree_path = settings.trees_dir / f"{doc_id}.json"

    # 4. Check if we've seen this exact document before
    cached = tree_path.exists()

    # 5. Save PDF if not already stored
    if not pdf_path.exists():
        pdf_path.write_bytes(contents)

    # 6. Create job record
    job_id = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        # Upsert document record
        await db.execute("""
            INSERT OR IGNORE INTO documents (doc_id, filename, file_path, tree_path)
            VALUES (?, ?, ?, ?)
        """, (doc_id, file.filename, str(pdf_path), str(tree_path) if cached else None))

        # Create job
        status = "complete" if cached else "received"
        await db.execute("""
            INSERT INTO jobs (job_id, doc_id, filename, status, progress)
            VALUES (?, ?, ?, ?, ?)
        """, (job_id, doc_id, file.filename, status, 100 if cached else 0))
        await db.commit()

    # If this is a fresh document, kick off background processing.
    # The task runs *after* this response is sent to the client.
    if not cached:
        background_tasks.add_task(process_document, job_id, doc_id, pdf_path, file.filename)
    return UploadResponse(
        job_id=job_id,
        doc_id=doc_id,
        filename=file.filename,
        status=status,
        cached=cached
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    return JobStatus(**dict(row))




@app.get("/tree/{doc_id}")
async def get_tree(doc_id: str):
    """
    Return the parsed PageIndex tree for a document.

    The doc_id is the SHA-256 hash of the original PDF — this is the
    same value returned by /upload as `doc_id`.
    """
    tree_path = settings.trees_dir / f"{doc_id}.json"
    if not tree_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Tree not found. Either the document was never uploaded, or processing hasn't finished yet."
        )

    return json.loads(tree_path.read_text(encoding="utf-8"))


@app.get("/documents")
async def list_documents():
    """
    Return all documents that have been uploaded.

    Includes whether each document has a tree built (i.e., processing finished).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT doc_id, filename, page_count,
                   tree_path IS NOT NULL AS has_tree,
                   created_at
            FROM documents
            ORDER BY created_at DESC
        """)
        rows = await cursor.fetchall()

    return [dict(row) for row in rows]


@app.post("/query")
async def query_document(req: QueryRequest):
    """
    Run a query against a previously-indexed document.

    Caching: hierarchical and long_context results are cached separately.
    Reported token counts come from real OpenAI usage data (input + output).
    """
    if req.mode not in ("hierarchical", "long_context", "both"):
        raise HTTPException(400, "mode must be one of: hierarchical, long_context, both")

    tree_path = settings.trees_dir / f"{req.doc_id}.json"
    pdf_path = settings.pdfs_dir / f"{req.doc_id}.pdf"

    if not tree_path.exists():
        raise HTTPException(404, "Document not indexed. Has processing finished?")
    if not pdf_path.exists():
        raise HTTPException(404, "PDF for this document is missing.")

    overall_start = time.time()
    result: dict = {
        "doc_id": req.doc_id,
        "query": req.query,
        "mode": req.mode,
    }

    hierarchical_from_cache = False
    long_context_from_cache = False
    tree_loaded = False
    tree = None

    # ---- Hierarchical component ----
    if req.mode in ("hierarchical", "both"):
        cached_h = get_component(req.doc_id, req.query, "hierarchical")
        if cached_h is not None:
            result["hierarchical"] = cached_h
            hierarchical_from_cache = True
        else:
            if not tree_loaded:
                tree = json.loads(tree_path.read_text(encoding="utf-8"))
                tree_loaded = True
            h_start = time.time()
            relevant_leaves, traversal_log, nav_usage = traverse_tree(req.query, tree)
            synthesis = synthesize_answer(req.query, relevant_leaves, pdf_path)

            # Total usage = navigation + synthesis
            total_usage = {
                "input_tokens": nav_usage["input_tokens"] + synthesis["usage"]["input_tokens"],
                "output_tokens": nav_usage["output_tokens"] + synthesis["usage"]["output_tokens"],
                "total_tokens": nav_usage["total_tokens"] + synthesis["usage"]["total_tokens"],
            }

            h_payload = {
                "answer": synthesis["answer"],
                "input_tokens": total_usage["input_tokens"],
                "output_tokens": total_usage["output_tokens"],
                "total_tokens": total_usage["total_tokens"],
                "leaves_used": synthesis["leaves_used"],
                "traversal_log": traversal_log,
                "nodes_explored": len(traversal_log),
                "nodes_kept": sum(
                    1 for e in traversal_log
                    if e["decision"] in ("explore", "leaf_relevant", "depth_capped_leaf")
                ),
                "latency_ms": int((time.time() - h_start) * 1000),
            }
            result["hierarchical"] = h_payload
            set_component(req.doc_id, req.query, "hierarchical", h_payload)

    # ---- Long-context component ----
    if req.mode in ("long_context", "both"):
        cached_lc = get_component(req.doc_id, req.query, "long_context")
        if cached_lc is not None:
            result["long_context"] = cached_lc
            long_context_from_cache = True
        else:
            lc_start = time.time()
            lc = long_context_answer(req.query, pdf_path)
            lc_payload = {
                "answer": lc["answer"],
                "input_tokens": lc["usage"]["input_tokens"],
                "output_tokens": lc["usage"]["output_tokens"],
                "total_tokens": lc["usage"]["total_tokens"],
                "skipped": lc["skipped"],
                "latency_ms": int((time.time() - lc_start) * 1000),
            }
            result["long_context"] = lc_payload
            set_component(req.doc_id, req.query, "long_context", lc_payload)

    # ---- Comparison block ----
    if req.mode == "both" and not result["long_context"].get("skipped"):
        h_tokens = result["hierarchical"]["total_tokens"]
        lc_tokens = result["long_context"]["total_tokens"]
        h_latency = result["hierarchical"]["latency_ms"]
        lc_latency = result["long_context"]["latency_ms"]

        if lc_tokens > 0 and h_tokens > 0:
            if h_tokens < lc_tokens:
                token_winner = "hierarchical"
                token_savings_pct = round((1 - h_tokens / lc_tokens) * 100, 1)
            else:
                token_winner = "long_context"
                token_savings_pct = round((1 - lc_tokens / h_tokens) * 100, 1)

            if h_latency < lc_latency:
                latency_winner = "hierarchical"
                latency_diff_ms = lc_latency - h_latency
            else:
                latency_winner = "long_context"
                latency_diff_ms = h_latency - lc_latency

            result["comparison"] = {
                "token_winner": token_winner,
                "token_savings_pct": token_savings_pct,
                "latency_winner": latency_winner,
                "latency_diff_ms": latency_diff_ms,
            }

    result["total_latency_ms"] = int((time.time() - overall_start) * 1000)

    # Cache status flags
    if req.mode == "hierarchical":
        result["from_cache"] = hierarchical_from_cache
    elif req.mode == "long_context":
        result["from_cache"] = long_context_from_cache
    else:
        result["from_cache"] = hierarchical_from_cache and long_context_from_cache

    if "hierarchical" in result:
        result["hierarchical"]["from_cache"] = hierarchical_from_cache
    if "long_context" in result:
        result["long_context"]["from_cache"] = long_context_from_cache

    return result