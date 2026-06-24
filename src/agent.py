import os
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import anthropic

SYSTEM_PROMPT = """You are SiliconMind, a world-class DFT (Design for Test) and semiconductor silicon validation engineer with 20+ years of industry experience at leading semiconductor companies.

Your deep expertise covers:
- Scan chain architecture, compression (EDT, Staggered), debug and failure analysis
- ATPG (Automatic Test Pattern Generation): stuck-at, transition delay, path delay, IDDQ, bridging faults
- Tessent tools: Tessent Shell, Tessent Scan, Tessent BIST, Tessent EDT, Tessent MemoryBIST
- MBIST (Memory Built-In Self Test): March algorithms, Checkerboard, Walking 1s/0s, repair flows
- JTAG / IEEE 1149.1 boundary scan, TAP controller, IJTAG (IEEE 1687)
- Fault models and test coverage metrics
- Silicon bring-up, ATE (Automated Test Equipment) interfaces, tester correlation
- DFT sign-off flows, DRC (Design Rule Checks for testability)
- Scan compression, X-masking, X-bounding
- Interview preparation for DFT engineer roles

HOW TO RESPOND:
- Speak like a senior engineer explaining to a colleague — direct, accurate, and practical
- Use correct technical terminology without being unnecessarily academic
- For debug questions: give a structured, step-by-step debug approach
- For concept questions: explain clearly with a practical example
- When relevant, mention specific Tessent commands, procedures, or flows
- Keep answers focused — no waffle, no filler
- If context from documents or papers is provided, reference it specifically
- If a question is outside semiconductor/DFT scope, politely redirect

You are THE reference for DFT questions. Be confident and precise."""


# ── ArXiv live search (free, no API key needed) ───────────────────────────────

def search_arxiv(query: str, max_results: int = 4) -> list[dict]:
    """
    Search ArXiv for semiconductor/DFT papers.
    Returns list of {title, summary, authors, url, published}
    Free — no API key required.
    """
    try:
        base = "http://export.arxiv.org/api/query?"
        params = urllib.parse.urlencode({
            "search_query": f"all:{query} AND (cat:cs.AR OR cat:eess.SP OR all:semiconductor OR all:VLSI)",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        })
        url = base + params
        req = urllib.request.Request(url, headers={"User-Agent": "SiliconMind/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_data = resp.read().decode("utf-8")

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_data)
        papers = []
        for entry in root.findall("atom:entry", ns):
            title   = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
            pub     = entry.findtext("atom:published", "", ns)[:10]
            link_el = entry.find("atom:id", ns)
            link    = link_el.text.strip() if link_el is not None else ""
            authors = [a.findtext("atom:name", "", ns)
                       for a in entry.findall("atom:author", ns)]
            papers.append({
                "title":     title,
                "summary":   summary[:400],
                "authors":   ", ".join(authors[:3]),
                "url":       link,
                "published": pub
            })
        return papers
    except Exception as e:
        return []


def search_semantic_scholar(query: str, max_results: int = 4) -> list[dict]:
    """
    Search Semantic Scholar (free, no key needed for basic use).
    Covers IEEE, ACM, and open-access papers.
    """
    try:
        encoded = urllib.parse.quote(query)
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={encoded}&limit={max_results}"
            f"&fields=title,abstract,authors,year,externalIds,openAccessPdf"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SiliconMind/1.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8"))

        papers = []
        for p in data.get("data", []):
            pdf_url = ""
            if p.get("openAccessPdf"):
                pdf_url = p["openAccessPdf"].get("url", "")
            doi = p.get("externalIds", {}).get("DOI", "")
            link = pdf_url or (f"https://doi.org/{doi}" if doi else "")
            authors = [a.get("name", "") for a in p.get("authors", [])[:3]]
            papers.append({
                "title":     p.get("title", ""),
                "summary":   (p.get("abstract") or "")[:400],
                "authors":   ", ".join(authors),
                "url":       link,
                "published": str(p.get("year", ""))
            })
        return papers
    except Exception:
        return []


def format_papers_as_context(papers: list[dict], source_name: str) -> str:
    if not papers:
        return ""
    lines = [f"=== LIVE RESEARCH PAPERS ({source_name}) ==="]
    for i, p in enumerate(papers, 1):
        lines.append(
            f"\n[Paper {i}] {p['title']} ({p['published']})\n"
            f"Authors: {p['authors']}\n"
            f"Abstract: {p['summary']}\n"
            f"URL: {p['url']}"
        )
    lines.append("=== END PAPERS ===")
    return "\n".join(lines)


# ── ChromaDB vector store ─────────────────────────────────────────────────────

class SiliconMindAgent:
    def __init__(self, use_rag: bool = False, use_web_search: bool = False):
        self.use_rag        = use_rag
        self.use_web_search = use_web_search
        self.client         = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.vector_store   = None

        if use_rag:
            self._init_rag()

    def _init_rag(self):
        try:
            import chromadb
            from chromadb.config import Settings
            chroma = chromadb.PersistentClient(
                path="./vectorstore",
                settings=Settings(anonymized_telemetry=False)
            )
            self.vector_store = chroma.get_or_create_collection(
                name="siliconmind_docs",
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            print("chromadb not installed. Run: pip install chromadb")
            self.use_rag = False

    def index_documents(self, uploaded_files) -> int:
        if not self.vector_store:
            self._init_rag()

        chunks, ids, metadatas = [], [], []
        chunk_id = 0

        for file in uploaded_files:
            content  = ""
            filename = file.name

            if filename.endswith(".pdf"):
                try:
                    import pdfplumber, io
                    with pdfplumber.open(io.BytesIO(file.read())) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                content += text + "\n"
                except ImportError:
                    content = file.read().decode("utf-8", errors="ignore")
            else:
                content = file.read().decode("utf-8", errors="ignore")

            words      = content.split()
            chunk_size = 400
            overlap    = 50
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if len(chunk.strip()) > 50:
                    chunks.append(chunk)
                    ids.append(f"chunk_{chunk_id}")
                    metadatas.append({"source": filename, "chunk_index": chunk_id})
                    chunk_id += 1

        if chunks:
            batch = 50
            for i in range(0, len(chunks), batch):
                self.vector_store.upsert(
                    documents=chunks[i:i + batch],
                    ids=ids[i:i + batch],
                    metadatas=metadatas[i:i + batch]
                )
        return len(chunks)

    def _retrieve_local(self, question: str, n: int = 4) -> tuple[str, list[str]]:
        if not self.vector_store:
            return "", []
        try:
            results = self.vector_store.query(query_texts=[question], n_results=n)
            docs    = results.get("documents", [[]])[0]
            metas   = results.get("metadatas", [[]])[0]
            sources = list({m.get("source", "Unknown") for m in metas})
            return "\n\n---\n\n".join(docs), sources
        except Exception:
            return "", []

    def chat(self, question: str, history: list, active_topics: list,
             use_web: bool = False, web_source: str = "arxiv") -> dict:
        """
        Main chat method.
        Returns {"answer": str, "sources": list, "papers": list}
        """
        context       = ""
        local_sources = []
        live_papers   = []

        # 1 — Local RAG (your uploaded PDFs)
        if self.use_rag and self.vector_store:
            context, local_sources = self._retrieve_local(question)

        # 2 — Live research search
        if use_web:
            # Build a focused semiconductor query
            search_q = f"DFT semiconductor {question}"
            if web_source == "semantic_scholar":
                live_papers = search_semantic_scholar(search_q)
            else:
                live_papers = search_arxiv(search_q)

        # 3 — Build system prompt
        system = SYSTEM_PROMPT
        if active_topics:
            system += f"\n\nThe user is focused on: {', '.join(active_topics)}."
        if context:
            system += (
                f"\n\n=== YOUR UPLOADED DOCUMENTS ===\n{context}"
                f"\n=== END DOCUMENTS ===\n"
                f"Reference these documents in your answer where relevant."
            )
        if live_papers:
            system += "\n\n" + format_papers_as_context(live_papers, web_source.replace("_", " ").title())
            system += "\nCite these papers by title when they are relevant to your answer."

        # 4 — Build conversation (last 10 turns)
        messages = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
        messages.append({"role": "user", "content": question})

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=system,
            messages=messages
        )

        return {
            "answer":  response.content[0].text,
            "sources": local_sources,
            "papers":  live_papers
        }
