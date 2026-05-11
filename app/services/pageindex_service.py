"""
Wrapper around the PageIndex CLI tool.

PageIndex is not a Python library — it's a standalone script we shell out to.
This module's job is to:
1. Invoke `run_pageindex.py` as a subprocess against a given PDF
2. Find the JSON output it produced
3. Return that JSON as a Python dict
4. Move the JSON into our project's `data/trees/` directory keyed by doc_id
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path
import re
import pymupdf

from app.config import settings

# Where we cloned PageIndex
PAGEINDEX_DIR = Path("vendor/PageIndex").resolve()
PAGEINDEX_RESULTS_DIR = PAGEINDEX_DIR / "results"


class PageIndexError(Exception):
    """Raised when PageIndex fails to produce a tree."""
    pass


def build_tree(
    pdf_path: Path,
    doc_id: str,
    original_filename: str = "",
    model: str = "gpt-5.4",
) -> dict:
    """
    Run PageIndex on the given PDF and return the resulting tree as a dict.

    The tree JSON is also persisted to `data/trees/<doc_id>.json`.

    Args:
        pdf_path: Absolute path to the PDF file (in data/pdfs/<doc_id>.pdf)
        doc_id: SHA-256 hash of the PDF — used as the canonical filename
        original_filename: Original name of the uploaded file (used as a
            fallback for the displayed title if the PDF has no embedded title)
        model: OpenAI model to pass to PageIndex

    Returns:
        The parsed tree dict (with keys "doc_name" and "structure")

    Raises:
        PageIndexError: if the subprocess fails or produces no output
    """
    if not pdf_path.is_absolute():
        pdf_path = pdf_path.resolve()

    if not pdf_path.exists():
        raise PageIndexError(f"PDF not found at {pdf_path}")

    # PageIndex names its output after the PDF's stem (filename without extension).
    # Since we save PDFs as <doc_id>.pdf, the output will be <doc_id>_structure.json.
    expected_output = PAGEINDEX_RESULTS_DIR / f"{pdf_path.stem}_structure.json"

    # Clear any stale output from a previous run with the same doc_id
    if expected_output.exists():
        expected_output.unlink()

    # Build the subprocess command.
    # We use sys.executable so we're guaranteed to use the same Python interpreter
    # that's running our FastAPI app — which has all the deps installed.
    cmd = [
        sys.executable,
        "run_pageindex.py",
        "--pdf_path", str(pdf_path),
        "--model", model,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=PAGEINDEX_DIR,         # critical: PageIndex's `from pageindex import *` needs this
            capture_output=True,        # capture stdout/stderr for logging
            text=True,
            timeout=1200,                # 20-minute hard ceiling
            check=False,                # we'll inspect returncode manually
        )
    except subprocess.TimeoutExpired as e:
        raise PageIndexError(f"PageIndex timed out: {e}") from e

    if result.returncode != 0:
        # PageIndex failed. Surface its stderr for debugging.
        raise PageIndexError(
            f"PageIndex exited with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if not expected_output.exists():
        raise PageIndexError(
            f"PageIndex finished but no output file at {expected_output}\n"
            f"STDOUT:\n{result.stdout}"
        )

    # Read the tree
    tree = json.loads(expected_output.read_text(encoding="utf-8"))

    # Replace doc_name with a human-readable title:
    # 1. PDF's embedded Title metadata, if any
    # 2. The original uploaded filename (without .pdf), if provided
    # 3. The PDF's filename stem (hash, last resort)
    tree["doc_name"] = _extract_display_title(pdf_path, original_filename)

    # Move it into our canonical location: data/trees/<doc_id>.json
    final_path = settings.trees_dir / f"{doc_id}.json"
    shutil.move(str(expected_output), str(final_path))

    # Persist the updated doc_name to disk (shutil.move just moved the original file)
    final_path.write_text(json.dumps(tree, indent=2), encoding="utf-8")

    return tree





def _normalize_for_comparison(s: str) -> str:
    """Strip everything except lowercase alphanumerics for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _titles_likely_match(metadata_title: str, filename_stem: str) -> bool:
    """
    Decide if the metadata title and filename describe the same document.

    Returns True only when the filename's normalized characters are mostly
    contained in the metadata's normalized characters — i.e. the metadata
    title looks like a cleaner version of the same name.
    """
    meta_norm = _normalize_for_comparison(metadata_title)
    name_norm = _normalize_for_comparison(filename_stem)

    if not meta_norm or not name_norm:
        return False

    # How much of the filename's characters appear in the metadata?
    # We require ≥70% overlap. This catches abbreviated filenames
    # ("berkshire_2024_ar" → "Berkshire 2024 Annual Report") while
    # rejecting unrelated metadata ("printmgr file" vs "berkshire").
    matched = sum(1 for ch in name_norm if ch in meta_norm)
    overlap = matched / len(name_norm)

    return overlap >= 0.7


def _extract_display_title(pdf_path: Path, original_filename: str = "") -> str:
    """
    Return a human-readable title for a PDF.

    Strategy:
      1. If PDF metadata has a Title AND it clearly corresponds to the
         filename (≥70% character overlap), use the metadata title —
         it's likely the cleaner human-readable version.
      2. Otherwise, fall back to the original filename without .pdf.
    """
    # Strip .pdf extension from the filename for our display fallback
    filename_stem = original_filename
    if filename_stem.lower().endswith(".pdf"):
        filename_stem = filename_stem[:-4]

    # Try metadata title only if it appears to match the filename
    try:
        doc = pymupdf.open(str(pdf_path))
        try:
            metadata_title = (doc.metadata or {}).get("title", "").strip()
            if metadata_title and _titles_likely_match(metadata_title, filename_stem):
                return metadata_title
        finally:
            doc.close()
    except Exception:
        pass

    return filename_stem