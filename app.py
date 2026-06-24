import streamlit as st
import os
from src.orchestrator import Orchestrator
from src.ingestion import extract_text, extract_from_url, smart_chunk, index_to_chromadb
from src.retrieval import init_vectorstore, retrieve
from src.feedback import log_feedback, get_stats

st.set_page_config(page_title="SiliconMind", page_icon="🔬", layout="wide")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0F6E56 0%, #1D9E75 100%);
    padding: 1.5rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem; color: white;
}
.agent-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; margin-right: 6px;
}
.badge-atpg       { background:#FFF3CD; color:#856404; }
.badge-mbist      { background:#D1ECF1; color:#0C5460; }
.badge-scan_debug { background:#F8D7DA; color:#721C24; }
.badge-jtag       { background:#D4EDDA; color:#155724; }
.badge-research   { background:#E2D9F3; color:#432874; }
.badge-general    { background:#E1F5EE; color:#0F6E56; }
.tool-chip {
    display: inline-block; background:#f0f2f6;
    border: 1px solid #dee2e6; padding: 2px 8px;
    border-radius: 10px; font-size: 11px; color:#495057; margin-right:4px;
}
.conf-high   { background:#D4EDDA; color:#155724; padding:4px 10px; border-radius:8px; font-size:12px; }
.conf-medium { background:#FFF3CD; color:#856404; padding:4px 10px; border-radius:8px; font-size:12px; }
.conf-low    { background:#F8D7DA; color:#721C24; padding:4px 10px; border-radius:8px; font-size:12px; }
.conf-none   { background:#e9ecef; color:#495057; padding:4px 10px; border-radius:8px; font-size:12px; }
.paper-card {
    background:#f8f9fa; border-left:3px solid #1D9E75;
    padding:10px 14px; border-radius:6px; margin:6px 0; font-size:13px;
}
.paper-card a { color:#0F6E56; font-weight:600; text-decoration:none; }
.paper-card a:hover { text-decoration:underline; }
.script-box {
    background:#1e1e1e; color:#d4d4d4; padding:16px;
    border-radius:8px; font-family:monospace; font-size:12px;
    overflow-x:auto; white-space:pre; margin:8px 0;
}
/* Consistent message styling */
[data-testid="stChatMessage"] table {
    width:100%; border-collapse:collapse;
    font-size:13px; margin:10px 0;
}
[data-testid="stChatMessage"] th {
    background:#E1F5EE; color:#0F6E56;
    padding:8px 12px; text-align:left;
    font-weight:600; border:1px solid #c3e6d8;
}
[data-testid="stChatMessage"] td {
    padding:7px 12px; border:1px solid #e0e0e0;
    font-size:13px; vertical-align:top;
}
[data-testid="stChatMessage"] tr:nth-child(even) td {
    background:#f8fffe;
}
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    font-size:15px; font-weight:600;
    color:#0F6E56; margin:14px 0 6px;
    border-bottom:1px solid #E1F5EE; padding-bottom:4px;
}
[data-testid="stChatMessage"] code {
    background:#f0f7f4; color:#0F6E56;
    padding:2px 6px; border-radius:4px;
    font-size:12px; font-family:monospace;
}
[data-testid="stChatMessage"] hr {
    border:none; border-top:1px solid #E1F5EE; margin:12px 0;
}
[data-testid="stChatMessage"] p {
    font-size:14px; line-height:1.6; margin:6px 0;
}
</style>
""", unsafe_allow_html=True)

AGENT_LABELS = {
    "atpg":       ("⚡ ATPG Agent",      "badge-atpg"),
    "mbist":      ("🧠 MBIST Agent",      "badge-mbist"),
    "scan_debug": ("🔍 Scan Debug Agent", "badge-scan_debug"),
    "jtag":       ("🔌 JTAG Agent",       "badge-jtag"),
    "research":   ("📰 Research Agent",   "badge-research"),
    "general":    ("🔬 SiliconMind",      "badge-general"),
}

CONF_CSS = {"high":"conf-high","medium":"conf-medium","low":"conf-low","none":"conf-none"}

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0;font-size:1.6rem;">🔬 SiliconMind v2.0</h2>
    <p style="margin:4px 0 0;opacity:0.85;font-size:0.9rem;">
        Multi-Agent DFT & Semiconductor Validation AI · LD7237 · Northumbria University
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.markdown("### 🤖 Agent Architecture")
    st.markdown("""
    <div style='font-size:12px;line-height:1.8;'>
    <b>Orchestrator</b> routes to:<br>
    <span class='agent-badge badge-atpg'>⚡ ATPG</span>Fault models, coverage<br>
    <span class='agent-badge badge-mbist'>🧠 MBIST</span>Memory test, repair<br>
    <span class='agent-badge badge-scan_debug'>🔍 Scan Debug</span>Chain failures, bring-up<br>
    <span class='agent-badge badge-jtag'>🔌 JTAG</span>Boundary scan, TAP<br>
    <span class='agent-badge badge-research'>📰 Research</span>ArXiv, Semantic Scholar<br>
    <span class='agent-badge badge-general'>🔬 General</span>Cross-domain, interview
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📚 Knowledge Base (RAG)")
    use_rag  = st.toggle("Use uploaded documents", value=False)
    doc_tabs = st.tabs(["📄 Files", "🔗 URL"])

    with doc_tabs[0]:
        uploaded = st.file_uploader("PDF, TXT, DOCX, PPTX",
                                     type=["pdf","txt","docx","pptx"],
                                     accept_multiple_files=True)
        if uploaded and st.button("📥 Index Documents", use_container_width=True):
            vs = init_vectorstore()
            total = 0
            with st.spinner("Indexing..."):
                for f in uploaded:
                    text, fname = extract_text(f)
                    chunks = smart_chunk(text, fname)
                    total += index_to_chromadb(chunks, vs)
            st.session_state.indexed = True
            st.success(f"✅ {total} chunks indexed")

    with doc_tabs[1]:
        url_input = st.text_input("ArXiv or paper URL",
                                   placeholder="https://arxiv.org/abs/...")
        if st.button("➕ Add URL", use_container_width=True) and url_input:
            vs = init_vectorstore()
            with st.spinner("Fetching..."):
                text, label = extract_from_url(url_input)
                if text:
                    chunks = smart_chunk(text, label)
                    n = index_to_chromadb(chunks, vs)
                    st.session_state.indexed = True
                    st.success(f"✅ {n} chunks added")
                else:
                    st.error("Could not fetch URL")

    if st.session_state.get("indexed"):
        st.info("📦 Documents ready")

    st.divider()
    st.markdown("### 🎯 Active Topics")
    topic_list    = ["Scan Chain / EDT","ATPG / Fault Models","MBIST",
                     "JTAG / Boundary Scan","Silicon Bring-up","Interview Prep"]
    active_topics = [t for t in topic_list if st.checkbox(t, value=True)]

    st.divider()
    # Feedback stats
    stats = get_stats()
    if stats["total_ratings"] > 0:
        st.markdown("### 📊 Answer Quality")
        st.metric("Satisfaction", f"{stats['satisfaction_pct']}%",
                  f"{stats['total_ratings']} ratings")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style='font-size:11px;color:#888;margin-top:1rem;'>
    v2.0 · Multi-agent · Tool use · RAG<br>
    Confidence scoring · Feedback loop<br>
    Claude Sonnet · ChromaDB · ArXiv<br>
    LD7237 · MSc · Northumbria University
    </div>""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "orchestrator" not in st.session_state: st.session_state.orchestrator = None

# ── Suggestions ───────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**💡 Try asking — each routes to a different specialist agent:**")
    cols = st.columns(2)
    suggestions = [
        "What causes scan chain failures in silicon bring-up?",
        "Calculate fault coverage: 95000 detected out of 98000 stuck-at faults",
        "Compare fault models for a 7nm automotive SoC",
        "Generate a Tessent script for scan insertion",
        "Walk me through the MBIST March C- algorithm",
        "Parse this Tessent report: Fault Coverage = 96.45%, Aborted = 234",
        "Find latest research papers on EDT scan compression",
        "Give me a JTAG TAP controller debug checklist",
    ]
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, use_container_width=True, key=f"sug_{i}"):
            st.session_state["pending_prompt"] = s
            st.rerun()

# ── Helper: render one assistant message ─────────────────────────
def render_assistant_msg(msg, idx):
    label, css = AGENT_LABELS.get(msg.get("agent","general"), ("🔬 SiliconMind","badge-general"))
    tools_html  = "".join(
        f'<span class="tool-chip">🔧 {t}</span>'
        for t in msg.get("tools_used", [])
    )
    st.markdown(f'<span class="agent-badge {css}">{label}</span>{tools_html}',
                unsafe_allow_html=True)

    # Confidence badge (Sprint 2 — from ORAssistant)
    conf = msg.get("confidence", {})
    if conf and conf.get("level") and conf["level"] != "none":
        conf_css = CONF_CSS.get(conf["level"], "conf-none")
        icons = {"high":"🟢","medium":"🟡","low":"🔴","none":"⚪"}
        icon  = icons.get(conf["level"], "⚪")
        st.markdown(f'<span class="{conf_css}">{icon} {conf["label"]}</span>',
                    unsafe_allow_html=True)

    # Answer
    answer = msg["content"]
    # If answer contains a Tessent script, render it nicely
    if "set_context" in answer or "read_verilog" in answer or "set_fault_type" in answer:
        parts = answer.split("```")
        for j, part in enumerate(parts):
            if j % 2 == 0:
                if part.strip():
                    st.markdown(part)
            else:
                code = part.replace("tcl\n","").replace("tcl","")
                st.markdown(f'<div class="script-box">{code}</div>', unsafe_allow_html=True)
    else:
        st.markdown(answer)

    # Sources from RAG
    if msg.get("sources"):
        st.markdown("**📄 From your docs:** " + " ".join(
            f'<span style="background:#E1F5EE;color:#0F6E56;padding:2px 8px;'
            f'border-radius:10px;font-size:11px;margin-right:4px;">{s}</span>'
            for s in msg["sources"]
        ), unsafe_allow_html=True)

    # Live papers
    if msg.get("papers"):
        with st.expander(f"📰 {len(msg['papers'])} research paper(s) used"):
            for p in msg["papers"]:
                link = (f'<a href="{p["url"]}" target="_blank">{p["title"]}</a>'
                        if p.get("url") else p.get("title",""))
                st.markdown(
                    f'<div class="paper-card">{link}<br>'
                    f'<span style="color:#666;">{p.get("authors","")} · '
                    f'{p.get("published","")} · {p.get("source","")}</span><br>'
                    f'{p.get("summary","")}</div>',
                    unsafe_allow_html=True
                )

    # Thumbs up/down (Sprint 2 — from ORAssistant)
    col1, col2, _ = st.columns([1, 1, 10])
    q = ""
    for m in reversed(st.session_state.messages[:idx]):
        if m["role"] == "user":
            q = m["content"]
            break
    if col1.button("👍", key=f"up_{idx}", help="Helpful"):
        log_feedback(q, msg.get("agent",""), msg["content"], 1,
                     msg.get("tools_used",[]), msg.get("confidence",{}).get("label",""))
        st.toast("Thanks for the feedback! 👍")
    if col2.button("👎", key=f"dn_{idx}", help="Not helpful"):
        log_feedback(q, msg.get("agent",""), msg["content"], -1,
                     msg.get("tools_used",[]), msg.get("confidence",{}).get("label",""))
        st.toast("Noted — we'll improve! 👎")


# ── Chat History ──────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🔬"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant_msg(msg, idx)

# ── Input ─────────────────────────────────────────────────────────
prompt = st.chat_input("Ask SiliconMind anything about DFT or semiconductor validation...")

# Pick up suggestion button clicks
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("⚠️ Enter your Anthropic API key in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🔬"):
        with st.spinner("SiliconMind thinking..."):
            if st.session_state.orchestrator is None:
                st.session_state.orchestrator = Orchestrator()

            # RAG retrieval with confidence scoring (Sprint 2)
            local_context, local_sources, confidence = "", [], {}
            if use_rag and st.session_state.get("indexed"):
                vs = init_vectorstore()
                local_context, local_sources, confidence = retrieve(prompt, vs)

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            result = st.session_state.orchestrator.run(
                question=prompt,
                history=history,
                active_topics=active_topics,
                local_context=local_context,
                local_sources=local_sources,
                confidence=confidence
            )

        new_msg = {
            "role":       "assistant",
            "content":    result["answer"],
            "agent":      result["agent"],
            "tools_used": result.get("tools_used", []),
            "papers":     result.get("papers", []),
            "sources":    result.get("sources", []),
            "confidence": result.get("confidence", {})
        }
        render_assistant_msg(new_msg, len(st.session_state.messages))
        st.session_state.messages.append(new_msg)
