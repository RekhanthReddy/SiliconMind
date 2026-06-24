# Contributing to SiliconMind

SiliconMind is an open community knowledge base for DFT and semiconductor validation engineers.
Anyone can contribute domain knowledge, documents, or code improvements.

---

## How to Contribute Documents

1. Fork this repository on GitHub
2. Add your PDF/TXT/DOCX files to the `community_docs/` folder
3. Update `community_docs/INDEX.md` with a one-line description
4. Submit a pull request with a short description of what you added

**What's welcome:**
- Textbook chapters on DFT, ATPG, MBIST, JTAG
- Open-source interview Q&A for DFT engineers
- Lab guides, debug checklists, bring-up notes
- IEEE open-access papers (link only — no copyrighted full text)
- Tool-agnostic methodology notes

**What's not welcome:**
- Proprietary tool documentation (Tessent, TetraMAX licensed content)
- NDA-covered internal company documents
- Anything with personal or confidential information

---

## How to Contribute Code

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make your changes — keep to the existing module structure
3. Test locally: `streamlit run app.py`
4. Submit a pull request describing what you changed and why

**Good code contributions:**
- New specialist agents (e.g. a Power-Aware DFT agent)
- New tools in `src/tools.py`
- Better chunking strategies in `src/ingestion.py`
- UI improvements in `app.py`
- Bug fixes

---

## Project Structure

```
siliconmind/
├── app.py                  ← Streamlit UI + main entry point
├── src/
│   ├── orchestrator.py     ← Routes questions to specialist agents
│   ├── tools.py            ← Callable tools (search, calculate, checklist)
│   ├── ingestion.py        ← Document parsing and chunking
│   └── retrieval.py        ← ChromaDB vector search
├── community_docs/         ← Contributed DFT knowledge (add yours here)
├── .streamlit/
│   └── config.toml         ← Theme config (safe to commit)
├── requirements.txt
└── README.md
```

---

## Questions?

Open a GitHub Issue — label it `question` and we'll help you get started.
