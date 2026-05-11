// ============================================================
// Visualized Hierarchical RAG — frontend logic
// ============================================================
//
// Talks to four backend endpoints:
//   POST /upload          — start indexing a PDF
//   GET  /status/{job_id} — poll processing progress
//   GET  /tree/{doc_id}   — fetch the tree structure
//   POST /query           — run a query, get answer + traversal log

// ----- Application state -----
const state = {
    docId: null,
    jobId: null,
    tree: null,
    pollHandle: null,
    treeRoot: null,           // d3.hierarchy root, persists across queries
    nodeStates: new Map(),    // node_id -> "default" | "skipped" | "explored" | "leaf"
};

// ----- Element refs -----
const el = {
    fileInput:   document.getElementById("file-input"),
    uploadBtn:   document.getElementById("upload-btn"),
    statusArea:  document.getElementById("status-area"),
    progressFill:document.getElementById("progress-fill"),
    statusText:  document.getElementById("status-text"),
    treeSection: document.getElementById("tree-section"),
    treeContainer:document.getElementById("tree-container"),
    querySection:document.getElementById("query-section"),
    queryInput:  document.getElementById("query-input"),
    modeSelect:  document.getElementById("mode-select"),
    queryBtn:    document.getElementById("query-btn"),
    answerArea:  document.getElementById("answer-area"),
};

// ============================================================
// Upload flow
// ============================================================

el.fileInput.addEventListener("change", () => {
    el.uploadBtn.disabled = el.fileInput.files.length === 0;
});

el.uploadBtn.addEventListener("click", uploadFile);

async function uploadFile() {
    const file = el.fileInput.files[0];
    if (!file) return;

    el.uploadBtn.disabled = true;
    el.statusArea.hidden = false;
    setProgress(0, "Uploading...");

    try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch("/upload", { method: "POST", body: formData });
        if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
        const data = await res.json();

        state.docId = data.doc_id;
        state.jobId = data.job_id;

        if (data.cached) {
            // Already processed before — go straight to tree
            setProgress(100, "Already indexed (cached)");
            await loadTree();
        } else {
            // Need to wait for background processing
            setProgress(5, "Processing started...");
            startPolling();
        }
    } catch (err) {
        showError(`Upload error: ${err.message}`);
        el.uploadBtn.disabled = false;
    }
}

function startPolling() {
    if (state.pollHandle) clearInterval(state.pollHandle);
    state.pollHandle = setInterval(checkStatus, 2000);
    checkStatus(); // run immediately too
}

async function checkStatus() {
    try {
        const res = await fetch(`/status/${state.jobId}`);
        if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
        const data = await res.json();

        setProgress(data.progress, `Status: ${data.status} (${data.progress}%)`);

        if (data.status === "complete") {
            clearInterval(state.pollHandle);
            state.pollHandle = null;
            await loadTree();
        } else if (data.status === "failed") {
            clearInterval(state.pollHandle);
            state.pollHandle = null;
            showError(`Processing failed: ${data.error || "unknown error"}`);
            el.uploadBtn.disabled = false;
        }
    } catch (err) {
        // Don't kill polling on transient errors — log and try again next tick
        console.warn("Polling hiccup:", err.message);
    }
}

function setProgress(pct, text) {
    el.progressFill.style.width = `${pct}%`;
    el.statusText.textContent = text;
}

// ============================================================
// Tree rendering (D3.js)
// ============================================================

async function loadTree() {
    try {
        const res = await fetch(`/tree/${state.docId}`);
        if (!res.ok) throw new Error(`Tree fetch failed: ${res.status}`);
        state.tree = await res.json();

        renderTree();
        el.treeSection.hidden = false;
        el.querySection.hidden = false;
    } catch (err) {
        showError(`Tree load error: ${err.message}`);
    }
}

function renderTree() {
    // PageIndex's tree has shape { doc_name, structure: [nodes...] }.
    // D3.hierarchy needs a single root, so we wrap with a synthetic root.
    const rootData = {
        title: state.tree.doc_name || "Document",
        node_id: "_root",
        nodes: state.tree.structure || [],
    };

    state.treeRoot = d3.hierarchy(rootData, d => d.nodes);

    // Start collapsed: hide all children below the root.
    state.treeRoot.children?.forEach(collapse);

    // Initialize every node's state to "default".
    state.nodeStates.clear();
    state.treeRoot.each(d => {
        state.nodeStates.set(d.data.node_id, "default");
    });

    drawTree();
}

function collapse(d) {
    if (d.children) {
        d._children = d.children;
        d._children.forEach(collapse);
        d.children = null;
    }
}

function drawTree() {
    // Preserve scroll position across re-renders
    const prevScrollLeft = el.treeContainer.scrollLeft;
    const prevScrollTop = el.treeContainer.scrollTop;

    // Clear previous render
    el.treeContainer.innerHTML = "";

    const containerWidth = el.treeContainer.clientWidth;

    // Compute the tree layout. nodeSize gives consistent spacing per node;
    // the SVG will grow to fit.
    const layout = d3.tree().nodeSize([28, 320]);
    layout(state.treeRoot);

    // Find layout extents to size the SVG.
    let minX = Infinity, maxX = -Infinity, maxY = 0;
    state.treeRoot.each(d => {
        if (d.x < minX) minX = d.x;
        if (d.x > maxX) maxX = d.x;
        if (d.y > maxY) maxY = d.y;
    });

    const width = Math.max(containerWidth, maxY + 250);
    const height = Math.max(400, (maxX - minX) + 80);
    const xOffset = -minX + 40;

    const svg = d3.select(el.treeContainer)
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    const g = svg.append("g")
        .attr("transform", `translate(60, ${xOffset})`);

    // Links (the lines between nodes)
    g.selectAll(".tree-link")
        .data(state.treeRoot.links())
        .enter()
        .append("path")
        .attr("class", "tree-link")
        .attr("d", d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x));

    // Nodes
    const node = g.selectAll(".tree-node")
        .data(state.treeRoot.descendants())
        .enter()
        .append("g")
        .attr("class", d => {
            const visualState = state.nodeStates.get(d.data.node_id) || "default";
            const hasKids = d._children ? "has-children" : "";
            return `tree-node state-${visualState} ${hasKids}`;
        })
        .attr("transform", d => `translate(${d.y},${d.x})`)
        .on("click", (event, d) => toggleNode(d));

    node.append("circle").attr("r", 6);

    // Tooltip on hover
    node.append("title")
        .text(d => {
            const summary = d.data.summary || "";
            const truncated = summary.length > 200
                ? summary.slice(0, 200) + "..."
                : summary;
            return `${d.data.title || "Untitled"}\n\n${truncated}`;
        });

    // Title text — truncated for readability
    node.append("text")
        .attr("dx", 10)
        .attr("dy", 4)
        .text(d => {
            const t = d.data.title || "Untitled";
            return t.length > 30 ? t.slice(0, 30) + "..." : t;
        });
        // Restore scroll position
    el.treeContainer.scrollLeft = prevScrollLeft;
    el.treeContainer.scrollTop = prevScrollTop;
}

function toggleNode(d) {
    if (d._children) {
        d.children = d._children;
        d._children = null;
    } else if (d.children) {
        d._children = d.children;
        d.children = null;
    } else {
        return; // leaf node — nothing to toggle
    }
    drawTree();
}

// ============================================================
// Query flow
// ============================================================

el.queryBtn.addEventListener("click", runQuery);
el.queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runQuery();
});

async function runQuery() {
    const query = el.queryInput.value.trim();
    if (!query || !state.docId) return;

    el.answerArea.innerHTML = `<div class="thinking">Thinking… (this can take 10–60 seconds)</div>`;
    el.queryBtn.disabled = true;

    try {
        const res = await fetch("/query", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                doc_id: state.docId,
                query: query,
                mode: el.modeSelect.value,
            }),
        });
        if (!res.ok) throw new Error(`Query failed: ${res.status}`);
        const data = await res.json();

        applyTraversalToTree(data);
        renderAnswer(data);
    } catch (err) {
        showError(`Query error: ${err.message}`);
    } finally {
        el.queryBtn.disabled = false;
    }
}

function applyTraversalToTree(data) {
    // Reset all nodes to default
    state.nodeStates.forEach((_, k) => state.nodeStates.set(k, "default"));

    const log = data.hierarchical?.traversal_log || [];
    const usedNodeIds = new Set(
        (data.hierarchical?.leaves_used || []).map(l => l.node_id)
    );

    log.forEach(entry => {
        let visualState;
        if (usedNodeIds.has(entry.node_id)) {
            visualState = "leaf";
        } else if (entry.decision === "skip") {
            visualState = "skipped";
        } else {
            visualState = "explored";
        }
        state.nodeStates.set(entry.node_id, visualState);
    });

    // Auto-expand any path that has explored/leaf nodes so the user can see them
    if (state.treeRoot) {
        state.treeRoot.each(d => {
            const s = state.nodeStates.get(d.data.node_id);
            if (s === "explored" || s === "leaf") {
                expandAncestors(d);
            }
        });
        drawTree();
    }
}

function expandAncestors(node) {
    let current = node.parent;
    while (current) {
        if (current._children) {
            current.children = current._children;
            current._children = null;
        }
        current = current.parent;
    }
}

function renderAnswer(data) {
    let html = "";
    // Show cached indicator at the top if applicable
    if (data.from_cache) {
        html += `<div class="cache-badge">⚡ Cached result (instant)</div>`;
    }
    if (data.hierarchical) {
        const h = data.hierarchical;
        const cacheTag = h.from_cache ? `<span class="cache-tag">⚡ cached</span>` : "";
        html += `
            <div class="answer-block">
                <h3>Hierarchical RAG ${cacheTag}</h3>
                <div class="answer-text">${escapeHtml(h.answer)}</div>
                <div class="stats">
                    <span>Tokens: ${h.total_tokens.toLocaleString()} (${h.input_tokens.toLocaleString()} in / ${h.output_tokens.toLocaleString()} out)</span>
                    <span>Latency: ${h.latency_ms} ms</span>
                    <span>Nodes explored: ${h.nodes_explored}</span>
                    <span>Nodes used in answer: ${h.leaves_used.length}</span>
                </div>
            </div>`;
    }

    if (data.long_context) {
        const lc = data.long_context;
        const cacheTag = lc.from_cache ? `<span class="cache-tag">⚡ cached</span>` : "";
        html += `
            <div class="answer-block">
                <h3>Long Context (baseline) ${cacheTag}</h3>
                <div class="answer-text">${escapeHtml(lc.answer)}</div>
                <div class="stats">
                    <span>Tokens: ${lc.total_tokens.toLocaleString()} (${lc.input_tokens.toLocaleString()} in / ${lc.output_tokens.toLocaleString()} out)</span>
                    <span>Latency: ${lc.latency_ms} ms</span>
                    ${lc.skipped ? "<span>(Skipped: too large)</span>" : ""}
                </div>
            </div>`;
    }

    if (data.comparison) {
        const c = data.comparison;
        const tokenWinnerLabel = c.token_winner === "hierarchical" ? "Hierarchical" : "Long Context";
        const latencyWinnerLabel = c.latency_winner === "hierarchical" ? "Hierarchical" : "Long Context";
        html += `
            <div class="comparison-block">
                <h3>Comparison</h3>
                <div>
                    <strong>${tokenWinnerLabel}</strong> used
                    <span class="metric">${c.token_savings_pct}%</span> fewer tokens.
                </div>
                <div style="margin-top:8px;">
                    <strong>${latencyWinnerLabel}</strong> was ${c.latency_diff_ms} ms faster.
                </div>
            </div>`;
    }

    el.answerArea.innerHTML = html;
}

// ============================================================
// Utilities
// ============================================================

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text || "";
    return div.innerHTML;
}

function showError(msg) {
    el.answerArea.innerHTML = `<div class="error">${escapeHtml(msg)}</div>`;
}