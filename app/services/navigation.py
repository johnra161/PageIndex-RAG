"""
Hierarchical retrieval over a PageIndex tree.

Given a query and a tree, we walk the tree top-down, asking the LLM at each
level which children are relevant. We collect the relevant leaf nodes, then
ask a stronger LLM to synthesize an answer from their content.

We also support a `long_context` mode that just stuffs the whole document
into a single LLM call — used as a baseline for comparison.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pymupdf

from app.services.llm_client import (
    NAVIGATION_MODEL,
    SYNTHESIS_MODEL,
    call_llm,
    count_tokens,
    empty_usage,
    add_usage,
)

# Bounded search to keep cost and latency predictable.
MAX_BREADTH = 20
MAX_DEPTH = 5
EMPTY_BATCH_PATIENCE = 2
MAX_CONTEXT_TOKENS = 200_000
LONG_CONTEXT_LIMIT = 200_000


# ---------- Tree utilities ----------

def _children(node: dict) -> list[dict]:
    return node.get("nodes", []) or []


def _is_leaf(node: dict) -> bool:
    return not _children(node)


def _node_label(node: dict) -> str:
    return node.get("title", node.get("node_id", "?"))


# ---------- Page text extraction ----------

def _extract_page_text(pdf_path: Path, start_page: int, end_page: int) -> str:
    doc = pymupdf.open(str(pdf_path))
    try:
        chunks = []
        end_page = min(end_page, doc.page_count)
        for page_num in range(start_page - 1, end_page):
            page = doc.load_page(page_num)
            chunks.append(page.get_text())
        return "\n\n".join(chunks)
    finally:
        doc.close()


# ---------- Navigation ----------

NAVIGATION_SYSTEM = (
    "You are a research assistant deciding which sections of a document are "
    "worth exploring to answer a user's question. A section is worth exploring "
    "if its title or summary suggests it MIGHT contain relevant information, "
    "OR if it likely contains subsections that might. When in doubt, mark it "
    "as relevant — exploring an irrelevant branch is cheap, but missing a "
    "relevant one means a worse answer."
)

NAVIGATION_PROMPT = """\
User question: {query}

Below are sections of a document. For each, decide whether it is likely to
contain information that helps answer the question.

Sections:
{sections}

Reply with ONLY a JSON object mapping each section's node_id to true or false.
Example: {{"0001": true, "0002": false, "0003": true}}
No explanation, no other text — just the JSON.
"""


def _format_sections_for_prompt(nodes: list[dict]) -> str:
    lines = []
    for node in nodes:
        nid = node.get("node_id", "?")
        title = node.get("title", "Untitled")
        summary = node.get("summary", "")
        if len(summary) > 800:
            summary = summary[:800] + "..."
        lines.append(f"- node_id={nid} | title: {title}\n  summary: {summary}")
    return "\n".join(lines)


def _parse_relevance_response(text: str, expected_ids: list[str]) -> dict[str, bool]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if not match:
        return {nid: False for nid in expected_ids}

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {nid: False for nid in expected_ids}

    result = {}
    for nid in expected_ids:
        val = parsed.get(nid, False)
        result[nid] = bool(val)
    return result


def _judge_relevance(query: str, candidates: list[dict]) -> tuple[dict[str, bool], dict]:
    #Returns (verdicts, usage).
    if not candidates:
        return {}, empty_usage()

    prompt = NAVIGATION_PROMPT.format(
        query=query,
        sections=_format_sections_for_prompt(candidates),
    )
    response, usage = call_llm(
        prompt,
        model=NAVIGATION_MODEL,
        system=NAVIGATION_SYSTEM,
        max_tokens=200,
    )
    expected_ids = [n.get("node_id", "?") for n in candidates]
    return _parse_relevance_response(response, expected_ids), usage




def traverse_tree(query: str, tree: dict) -> tuple[list[dict], list[dict], dict]:
    """
    Walk the tree top-down, collecting relevant leaf nodes.

    Returns:
        relevant_leaves, traversal_log, total_usage
    """
    structure = tree.get("structure", [])
    relevant_leaves: list[dict] = []
    traversal_log: list[dict] = []
    total_usage = empty_usage()

    def visit(siblings: list[dict], depth: int, path: list[str]) -> None:
        if not siblings or depth > MAX_DEPTH:
            return

        # Process siblings in batches of MAX_BREADTH with early stopping
        # after EMPTY_BATCH_PATIENCE consecutive batches yield no relevance.
        # Patience counter persists across batches within this single visit call;
        # any batch with at least one hit resets it.
        batch_start = 0
        empty_batches_seen = 0

        while batch_start < len(siblings):
            batch = siblings[batch_start:batch_start + MAX_BREADTH]
            batch_start += MAX_BREADTH

            verdicts, usage = _judge_relevance(query, batch)
            add_usage(total_usage, usage)

            relevant_in_batch = 0

            for node in batch:
                nid = node.get("node_id", "?")
                label = _node_label(node)
                current_path = path + [label]

                if verdicts.get(nid, False):
                    relevant_in_batch += 1

                    if _is_leaf(node):
                        # Case 1: true leaf node
                        relevant_leaves.append({"node": node, "path": current_path})
                        traversal_log.append({
                            "node_id": nid,
                            "title": label,
                            "path": current_path,
                            "decision": "leaf_relevant",
                            "depth": depth,
                        })
                    elif depth + 1 > MAX_DEPTH:
                        # Case 2: depth-capped fallback
                        relevant_leaves.append({"node": node, "path": current_path})
                        traversal_log.append({
                            "node_id": nid,
                            "title": label,
                            "path": current_path,
                            "decision": "depth_capped_leaf",
                            "depth": depth,
                        })
                    else:
                        # Case 3: recurse normally
                        traversal_log.append({
                            "node_id": nid,
                            "title": label,
                            "path": current_path,
                            "decision": "explore",
                            "depth": depth,
                        })
                        visit(_children(node), depth + 1, current_path)
                else:
                    traversal_log.append({
                        "node_id": nid,
                        "title": label,
                        "path": current_path,
                        "decision": "skip",
                        "depth": depth,
                    })

            # Early-stopping logic
            if relevant_in_batch == 0:
                empty_batches_seen += 1
                if empty_batches_seen >= EMPTY_BATCH_PATIENCE:
                    break
            else:
                empty_batches_seen = 0

    visit(structure, depth=0, path=[])
    return relevant_leaves, traversal_log, total_usage


# ---------- Synthesis ----------

SYNTHESIS_SYSTEM = (
    "You are a precise research assistant. Answer the user's question using "
    "ONLY the document sections provided. If the sections do not contain the "
    "answer, say so plainly. Cite which sections you used by their path "
    "(e.g. 'Section: Chairman's Letter > Operating Performance')."
)

SYNTHESIS_PROMPT = """\
Question: {query}

Document sections:
{context}

Answer the question above using only the sections provided. Cite which
sections support your answer.
"""


def synthesize_answer(
    query: str,
    relevant_leaves: list[dict],
    pdf_path: Path,
) -> dict[str, Any]:
    """
    Route between direct synthesis and map-reduce based on content size.

    Returns dict with: answer, leaves_used, usage, synthesis_strategy.
    """
    if not relevant_leaves:
        return {
            "answer": "No relevant sections of the document were found for this query.",
            "leaves_used": [],
            "usage": empty_usage(),
            "synthesis_strategy": "none",
        }

    total_estimated = 0
    leaf_data = []
    for item in relevant_leaves:
        node = item["node"]
        start_page = node.get("start_index", 1)
        end_page = node.get("end_index", start_page)
        text = _extract_page_text(pdf_path, start_page, end_page)
        tokens = count_tokens(text, model=SYNTHESIS_MODEL)
        total_estimated += tokens
        leaf_data.append({"item": item, "text": text, "tokens": tokens})

    if total_estimated <= MAX_CONTEXT_TOKENS:
        return _direct_synthesis(query, leaf_data)
    else:
        return _map_reduce_synthesis(query, leaf_data)


def _direct_synthesis(query: str, leaf_data: list[dict]) -> dict[str, Any]:
    parts = []
    total_context_tokens = 0
    included = []

    for ld in leaf_data:
        path_str = " > ".join(ld["item"]["path"])
        section_block = f"[Section: {path_str}]\n{ld['text']}\n"
        section_tokens = count_tokens(section_block, model=SYNTHESIS_MODEL)

        if total_context_tokens + section_tokens > MAX_CONTEXT_TOKENS:
            break

        parts.append(section_block)
        total_context_tokens += section_tokens
        included.append(ld["item"])

    context = "\n---\n".join(parts)

    answer, usage = call_llm(
        SYNTHESIS_PROMPT.format(query=query, context=context),
        model=SYNTHESIS_MODEL,
        system=SYNTHESIS_SYSTEM,
        max_tokens=1500,
        temperature=0.2,
    )

    return {
        "answer": answer,
        "leaves_used": [
            {"node_id": item["node"].get("node_id"), "path": item["path"]}
            for item in included
        ],
        "usage": usage,
        "synthesis_strategy": "direct",
    }


MAP_PROMPT = """\
The user asked: {query}

Below is one section of a document. Extract ONLY the parts of this section
that help answer the question. Be concise — preserve specific numbers, dates,
and quotes verbatim, but omit anything irrelevant.

If this section contains nothing relevant to the question, reply with the
single word: NONE

Section ({path}):
{text}
"""

REDUCE_PROMPT = """\
The user asked: {query}

Below are relevant extracts from multiple sections of a document. Synthesize
them into a complete, well-cited answer. Cite which sections support each
claim using the section paths in brackets.

Extracts:
{extracts}
"""


def _map_reduce_synthesis(query: str, leaf_data: list[dict]) -> dict[str, Any]:
    extracts = []
    total_usage = empty_usage()
    included = []

    for ld in leaf_data:
        path_str = " > ".join(ld["item"]["path"])
        truncated_text = ld["text"][:48_000]

        extract, usage = call_llm(
            MAP_PROMPT.format(query=query, path=path_str, text=truncated_text),
            model=NAVIGATION_MODEL,
            max_tokens=400,
            temperature=0.0,
        )
        add_usage(total_usage, usage)
        extract = extract.strip()

        if not extract or extract.upper().startswith("NONE"):
            continue

        extracts.append(f"[Section: {path_str}]\n{extract}")
        included.append(ld["item"])

    if not extracts:
        return {
            "answer": "After reviewing the relevant sections, none contained information that directly answers the question.",
            "leaves_used": [],
            "usage": total_usage,
            "synthesis_strategy": "map_reduce",
        }

    combined_extracts = "\n\n---\n\n".join(extracts)

    answer, reduce_usage = call_llm(
        REDUCE_PROMPT.format(query=query, extracts=combined_extracts),
        model=SYNTHESIS_MODEL,
        system=SYNTHESIS_SYSTEM,
        max_tokens=1500,
        temperature=0.2,
    )
    add_usage(total_usage, reduce_usage)

    return {
        "answer": answer,
        "leaves_used": [
            {"node_id": item["node"].get("node_id"), "path": item["path"]}
            for item in included
        ],
        "usage": total_usage,
        "synthesis_strategy": "map_reduce",
    }


# ---------- Long-context baseline ----------

def long_context_answer(query: str, pdf_path: Path) -> dict[str, Any]:
    doc = pymupdf.open(str(pdf_path))
    try:
        full_text = "\n\n".join(page.get_text() for page in doc)
    finally:
        doc.close()

    estimated_tokens = count_tokens(full_text, model=SYNTHESIS_MODEL)

    if estimated_tokens > LONG_CONTEXT_LIMIT:
        return {
            "answer": (
                f"Document is too large for long-context mode "
                f"({estimated_tokens:,} estimated tokens, limit {LONG_CONTEXT_LIMIT:,}). "
                f"Use hierarchical mode instead."
            ),
            "usage": empty_usage(),
            "skipped": True,
        }

    answer, usage = call_llm(
        f"Question: {query}\n\nDocument:\n{full_text}\n\nAnswer the question above using only the document.",
        model=SYNTHESIS_MODEL,
        system="You are a precise research assistant. Answer using only the document provided.",
        max_tokens=1500,
        temperature=0.2,
    )

    return {
        "answer": answer,
        "usage": usage,
        "skipped": False,
    }