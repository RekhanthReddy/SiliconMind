# 🔬 SiliconMind — Multi-Agent DFT & Semiconductor Validation AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://siliconmind.streamlit.app)

> A domain-specific multi-agent AI system for DFT and semiconductor validation engineers.
> Ask questions like a senior engineer answers them.

**LD7237 Contemporary Computing and Digital Technologies · MSc · Northumbria University**

---

## What It Does

SiliconMind routes your question to the right specialist agent automatically:

| Agent | Handles |
|---|---|
| ⚡ ATPG Agent | Fault models, coverage, test patterns, scan compression |
| 🧠 MBIST Agent | Memory test, March algorithms, repair, BIST architecture |
| 🔍 Scan Debug Agent | Scan chain failures, EDT, silicon bring-up, ATE correlation |
| 🔌 JTAG Agent | Boundary scan, TAP controller, IEEE 1149.1, IJTAG |
| 📰 Research Agent | Live papers from ArXiv and Semantic Scholar (IEEE) |
| 🔬 General Agent | Cross-domain, interview prep, methodology |

**Built-in tools the agents can call:**
- 🔧 `search_arxiv` — live semiconductor research papers (free, no key needed)
- 🔧 `search_semantic_scholar` — IEEE + open-access papers
- 🔧 `calculate_fault_coverage` — computes coverage %, grades it, gives recommendations
- 🔧 `generate_debug_checklist` — structured checklists for scan, ATPG, MBIST, JTAG, bring-up
- 🔧 `fetch_paper_summary` — summarise any ArXiv paper from a URL

---

## Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/RekhanthReddy/siliconmind.git
cd siliconmind

# 2. Create conda environment
conda create -n siliconmind python=3.12 -y
conda activate siliconmind

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Get a free API key at **console.anthropic.com** and paste it in the sidebar.

---

## Deploy to Streamlit Cloud (Free, Public URL)

1. Push this repo to GitHub
2. Go to **share.streamlit.io** → New app → select your repo → `app.py`
3. Under **Advanced settings → Secrets**, add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
4. Click Deploy — live in ~2 minutes

---

## Architecture

```
User Question
      │
      ▼
 Orchestrator  ─── classifies intent ──► Specialist Agent
      │                                        │
      │                                   Uses Tools:
      │                                   • search_arxiv
      │                                   • search_semantic_scholar
      │                                   • calculate_fault_coverage
      │                                   • generate_debug_checklist
      │                                   • fetch_paper_summary
      │
      ├── ChromaDB (your uploaded PDFs)  ← RAG layer
      └── ArXiv / Semantic Scholar       ← Live research layer
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — add your DFT documents to `community_docs/` and open a PR.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude Sonnet (Anthropic) with tool use |
| Vector DB | ChromaDB (local persistent) |
| Document parsing | pdfplumber, python-docx, python-pptx |
| Live research | ArXiv API, Semantic Scholar API |
| UI | Streamlit |
| Language | Python 3.12 |
