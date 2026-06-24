"""
SiliconMind Orchestrator
────────────────────────
Routes incoming questions to the right specialist sub-agent.
Uses Claude to classify the intent, then dispatches accordingly.
"""

import os
import json
import anthropic
from .tools import TOOLS, execute_tool

# ── Sub-agent system prompts ──────────────────────────────────────────────────

AGENT_PROMPTS = {
    "atpg": """You are the ATPG Specialist Agent within SiliconMind.
Your sole focus: Automatic Test Pattern Generation, fault models, and test coverage.

Deep expertise:
- Fault models: stuck-at, transition delay, path delay, IDDQ, cell-aware, bridging
- ATPG engines: Tessent FastScan, TetraMAX, scan pattern generation
- Fault coverage analysis, abort analysis, compaction
- X-masking, X-bounding, X-propagation
- EDT scan compression, channel masking
- ATPG constraints: clock gating, tristate, async pins

Use tools when: user asks for papers, coverage calculations, or debug checklists.
Be precise. Give Tessent commands where relevant. Think like a senior ATPG engineer.""",

    "mbist": """You are the MBIST Specialist Agent within SiliconMind.
Your sole focus: Memory Built-In Self Test, memory repair, and memory characterisation.

Deep expertise:
- MBIST algorithms: March C-, March LR, Checkerboard, Walking 1s/0s, Galloping
- Tessent MemoryBIST architecture, controller, and configuration
- Memory types: SRAM, ROM, CAM, register files, embedded DRAM
- Memory redundancy and repair: row/column repair, fuse programming
- Memory characterisation: retention, read/write margin, timing
- Hierarchical BIST, BIST infrastructure DFT

Use tools when: user asks for papers, debug checklists, or coverage metrics.
Be precise. Give Tessent MemoryBIST commands where relevant.""",

    "scan_debug": """You are the Scan Chain & Silicon Debug Specialist Agent within SiliconMind.
Your sole focus: scan chain architecture, EDT, silicon bring-up, and failure debug.

Deep expertise:
- Scan chain architecture: single, multiple, segmented chains
- EDT (Embedded Deterministic Test): compression/decompression logic
- Scan debug: shift failures, capture failures, clock domain issues
- Silicon bring-up: first power-on, tester bring-up, ATE correlation
- Failure analysis: chain diagnosis, fault localisation, physical debug
- DFT sign-off: DRC rules, scan insertion, timing closure

Use tools when: user asks for debug checklists, papers, or bring-up issues.
Provide structured debug flows. Think like an engineer at a lab bench.""",

    "jtag": """You are the JTAG & Boundary Scan Specialist Agent within SiliconMind.
Your sole focus: JTAG, boundary scan, and embedded instrument standards.

Deep expertise:
- IEEE 1149.1 JTAG: TAP controller state machine, instruction register, data registers
- Boundary scan: SAMPLE, PRELOAD, EXTEST, BYPASS, IDCODE instructions
- BSDL files, board-level interconnect test
- IEEE 1687 IJTAG: instrument access networks, SIB, ICL, PDL
- IEEE 1500 core test wrapper
- JTAG debug interfaces: ARM CoreSight, RISC-V debug

Use tools when: user asks for papers or debug checklists.
Be precise about TAP states and instruction encodings.""",

    "research": """You are the Research Agent within SiliconMind.
Your sole focus: finding, summarising, and explaining the latest DFT and semiconductor research.

You ALWAYS use the search_arxiv or search_semantic_scholar tools to find papers.
Never answer research questions from memory alone — always search first.

After searching:
- Summarise what the papers say about the topic
- Highlight the most relevant findings for a DFT engineer
- Cite papers by title and year
- Explain technical contributions in plain engineering terms""",

    "general": """You are SiliconMind, a senior DFT and semiconductor validation engineer.
You handle general questions that span multiple DFT domains or don't fit a single speciality.

Your expertise covers the full DFT stack: scan, ATPG, MBIST, JTAG, silicon bring-up,
ATE interfaces, DFT sign-off, and semiconductor validation flows.

Use tools when available and relevant. Be direct and practical."""
}

# ── Intent classifier ─────────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """You are a routing classifier for SiliconMind, a DFT engineering AI agent.

Classify the user's question into exactly ONE of these specialist agents:
- atpg        → ATPG, fault models, fault coverage, test patterns, scan compression, EDT
- mbist       → MBIST, memory test, March algorithms, memory repair, BIST
- scan_debug  → Scan chain failure, scan debug, silicon bring-up, ATE, tester bring-up
- jtag        → JTAG, boundary scan, TAP controller, IEEE 1149.1, IJTAG, IEEE 1687
- research    → "latest research", "papers on", "what's new in", "recent work"
- general     → Questions spanning multiple areas, career advice, interview prep, general DFT

Respond with ONLY a JSON object: {"agent": "<agent_name>", "confidence": <0.0-1.0>}
No explanation, no preamble."""


class Orchestrator:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def classify(self, question: str) -> str:
        """Route question to the right specialist agent."""
        try:
            resp = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=60,
                system=CLASSIFIER_PROMPT,
                messages=[{"role": "user", "content": question}]
            )
            text = resp.content[0].text.strip()
            data = json.loads(text)
            return data.get("agent", "general")
        except Exception:
            return "general"

    def run(self, question: str, history: list,
            active_topics: list, local_context: str = "",
            local_sources: list = None, confidence: dict = None) -> dict:
        """
        Full orchestration loop:
        1. Classify → pick specialist agent
        2. Run agent with tool_use loop
        3. Return answer + metadata
        """
        agent_name   = self.classify(question)
        system       = AGENT_PROMPTS.get(agent_name, AGENT_PROMPTS["general"])
        tool_results = []
        papers_used  = []

        # Inject local RAG context
        if local_context:
            system += (
                f"\n\n=== UPLOADED DOCUMENTS ===\n{local_context}"
                f"\n=== END DOCUMENTS ===\n"
                f"Reference these when relevant."
            )
        if active_topics:
            system += f"\n\nUser is focused on: {', '.join(active_topics)}."

        # Build message history (last 10 turns)
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in history[-10:]
        ]
        messages.append({"role": "user", "content": question})

        # ── Agentic tool-use loop ─────────────────────────────────────────────
        max_iterations = 8
        for _ in range(max_iterations):
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system=system,
                tools=TOOLS,
                messages=messages
            )

            # If Claude wants to use a tool
            if response.stop_reason == "tool_use":
                # Add assistant's response to message history
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                # Execute each tool Claude requested
                tool_result_content = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = execute_tool(block.name, block.input)

                        # Collect papers for UI display
                        if "papers" in result:
                            papers_used.extend(result["papers"])

                        tool_result_content.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     json.dumps(result)
                        })
                        tool_results.append({
                            "tool":   block.name,
                            "input":  block.input,
                            "output": result
                        })

                # Feed tool results back to Claude
                messages.append({
                    "role":    "user",
                    "content": tool_result_content
                })
                # Continue loop so Claude can respond after seeing tool results

            else:
                # Claude has finished — extract final text answer
                answer = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        answer += block.text
                break
        else:
            answer = "I searched multiple sources but couldn't find specific papers on that exact topic. Here's what I found:\n\n" + "\n".join([f"- {t['tool']}: {t['input']}" for t in tool_results[:3]]) + "\n\nTry rephrasing — for example: 'latest papers on scan chain compression' or 'EDT scan research 2024'."

        return {
            "answer":       answer,
            "agent":        agent_name,
            "tools_used":   [t["tool"] for t in tool_results],
            "tool_details": tool_results,
            "papers":       papers_used,
            "sources":      local_sources or [],
            "confidence":   confidence or {"level": "none", "label": "Answering from general knowledge", "score": 0.0, "from_docs": False}
        }
