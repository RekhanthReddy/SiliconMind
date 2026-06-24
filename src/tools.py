"""
SiliconMind Tools  — v2.0
─────────────────────────
Sprint 1: analyse_netlist, parse_fault_report  (stolen + improved from DFTAgent)
Sprint 2: confidence scoring in retrieve()     (stolen from ORAssistant)
Sprint 3: compare_fault_models, generate_tessent_script  (original, no competitor has these)
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import re
import os


# ═══════════════════════════════════════════════════════════════════
# TOOL SCHEMAS  (what Claude sees when deciding which tool to call)
# ═══════════════════════════════════════════════════════════════════

TOOLS = [

    # ── Existing ─────────────────────────────────────────────────
    {
        "name": "search_arxiv",
        "description": (
            "Search ArXiv for the latest semiconductor, DFT, VLSI, or EDA research papers. "
            "Use when the user asks about recent research, new techniques, or wants papers on a topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "e.g. 'scan compression EDT VLSI'"},
                "max_results": {"type": "integer", "description": "1-6", "default": 4}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_semantic_scholar",
        "description": (
            "Search Semantic Scholar for IEEE, ACM and open-access papers on DFT/semiconductor topics. "
            "Broader coverage than ArXiv — includes published IEEE conference and journal papers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "e.g. 'ATPG fault coverage optimisation'"},
                "max_results": {"type": "integer", "description": "1-6", "default": 4}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_fault_coverage",
        "description": (
            "Calculate fault coverage percentage, grade it, and give recommendations. "
            "Use when the user provides fault counts or asks about coverage metrics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "detected_faults":   {"type": "number", "description": "Number of detected faults"},
                "total_faults":      {"type": "number", "description": "Total number of faults"},
                "possibly_detected": {"type": "number", "description": "Possibly detected faults", "default": 0},
                "test_type":         {"type": "string", "description": "stuck-at / transition / IDDQ", "default": "stuck-at"}
            },
            "required": ["detected_faults", "total_faults"]
        }
    },
    {
        "name": "generate_debug_checklist",
        "description": (
            "Generate a structured step-by-step debug checklist for common DFT problems. "
            "Use when the user is debugging scan chain failures, low coverage, BIST failures, "
            "bring-up issues, or ATPG aborts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "problem_type":   {
                    "type": "string",
                    "description": "One of: scan_chain_failure, low_fault_coverage, mbist_failure, "
                                   "jtag_failure, silicon_bringup, atpg_abort"
                },
                "design_context": {
                    "type": "string",
                    "description": "Optional context e.g. '28nm, 500 scan chains, EDT 50x'",
                    "default": ""
                }
            },
            "required": ["problem_type"]
        }
    },
    {
        "name": "fetch_paper_summary",
        "description": "Fetch and summarise a research paper from an ArXiv URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full ArXiv URL"}
            },
            "required": ["url"]
        }
    },

    # ── Sprint 1: stolen + greatly improved from DFTAgent ─────────
    {
        "name": "analyse_netlist",
        "description": (
            "Parse and analyse a Verilog, BENCH, or SPICE netlist file. "
            "Extracts gate count, I/O ports, module hierarchy, and estimates ATPG complexity. "
            "Use when the user uploads or pastes a netlist, or asks about circuit structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "netlist_text": {
                    "type": "string",
                    "description": "Raw text content of the netlist file"
                },
                "file_format": {
                    "type": "string",
                    "description": "Format hint: verilog, bench, spice, or auto",
                    "default": "auto"
                }
            },
            "required": ["netlist_text"]
        }
    },
    {
        "name": "parse_fault_report",
        "description": (
            "Parse a Tessent, Atalanta, or TetraMAX fault report to extract coverage %, "
            "abort count, undetected faults, and CPU time. "
            "Use when the user pastes or uploads ATPG output logs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "report_text": {
                    "type": "string",
                    "description": "Raw text of the fault report / ATPG log"
                },
                "tool_hint": {
                    "type": "string",
                    "description": "Tool that generated the report: tessent, atalanta, tetramax, or auto",
                    "default": "auto"
                }
            },
            "required": ["report_text"]
        }
    },

    # ── Sprint 3: original tools — no competitor has these ────────
    {
        "name": "compare_fault_models",
        "description": (
            "Compare different fault models (stuck-at, transition delay, IDDQ, cell-aware, bridging) "
            "for a given design context. Recommends which models to run and in what order. "
            "Use when the user asks which fault model to use or how to plan their ATPG strategy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "design_type": {
                    "type": "string",
                    "description": "e.g. 'automotive SoC', 'mobile AP', 'FPGA', 'mixed-signal', 'memory chip'"
                },
                "process_node": {
                    "type": "string",
                    "description": "e.g. '7nm', '28nm', '65nm', '180nm'",
                    "default": "unknown"
                },
                "quality_target": {
                    "type": "string",
                    "description": "Target application: automotive_iso26262, consumer, aerospace, medical, general",
                    "default": "general"
                },
                "constraints": {
                    "type": "string",
                    "description": "Optional: time budget, tester limitations, pattern count limit",
                    "default": ""
                }
            },
            "required": ["design_type"]
        }
    },
    {
        "name": "generate_tessent_script",
        "description": (
            "Generate a ready-to-run Tessent Shell TCL script for a given DFT task. "
            "Use when the user asks for a Tessent script, command flow, or TCL template."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "One of: scan_insertion, atpg_stuck_at, atpg_transition, "
                                   "mbist_setup, edt_setup, fault_report, drc_check, scan_diagnosis"
                },
                "design_name": {
                    "type": "string",
                    "description": "Name of the design/top module",
                    "default": "my_design"
                },
                "options": {
                    "type": "string",
                    "description": "Optional extra context e.g. 'EDT compression 50x, 4 clock domains'",
                    "default": ""
                }
            },
            "required": ["task"]
        }
    }
]


# ═══════════════════════════════════════════════════════════════════
# EXISTING TOOL EXECUTORS
# ═══════════════════════════════════════════════════════════════════

def search_arxiv(query: str, max_results: int = 4) -> dict:
    try:
        params = urllib.parse.urlencode({
            "search_query": f"all:{query} AND (cat:cs.AR OR cat:eess.SP OR all:semiconductor OR all:VLSI)",
            "start": 0, "max_results": min(max_results, 6),
            "sortBy": "relevance", "sortOrder": "descending"
        })
        req = urllib.request.Request(
            "http://export.arxiv.org/api/query?" + params,
            headers={"User-Agent": "SiliconMind/2.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_data = resp.read().decode("utf-8")
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_data)
        papers = []
        for entry in root.findall("atom:entry", ns):
            title   = entry.findtext("atom:title",   "", ns).strip().replace("\n", " ")
            summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")[:400]
            pub     = entry.findtext("atom:published","", ns)[:10]
            link_el = entry.find("atom:id", ns)
            authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
            papers.append({
                "title": title, "summary": summary,
                "authors": ", ".join(authors[:3]),
                "url": link_el.text.strip() if link_el is not None else "",
                "published": pub, "source": "ArXiv"
            })
        return {"papers": papers, "count": len(papers), "source": "ArXiv"}
    except Exception as e:
        return {"papers": [], "error": str(e), "source": "ArXiv"}


def search_semantic_scholar(query: str, max_results: int = 4) -> dict:
    try:
        encoded = urllib.parse.quote(query)
        url = (f"https://api.semanticscholar.org/graph/v1/paper/search"
               f"?query={encoded}&limit={min(max_results,6)}"
               f"&fields=title,abstract,authors,year,externalIds,openAccessPdf")
        req = urllib.request.Request(url, headers={"User-Agent": "SiliconMind/2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        papers = []
        for p in data.get("data", []):
            pdf_url = (p.get("openAccessPdf") or {}).get("url", "")
            doi     = (p.get("externalIds") or {}).get("DOI", "")
            link    = pdf_url or (f"https://doi.org/{doi}" if doi else "")
            authors = [a.get("name","") for a in (p.get("authors") or [])[:3]]
            papers.append({
                "title":     p.get("title",""),
                "summary":   (p.get("abstract") or "")[:400],
                "authors":   ", ".join(authors),
                "url":       link,
                "published": str(p.get("year","")),
                "source":    "Semantic Scholar"
            })
        return {"papers": papers, "count": len(papers), "source": "Semantic Scholar"}
    except Exception as e:
        return {"papers": [], "error": str(e), "source": "Semantic Scholar"}


def calculate_fault_coverage(detected_faults: float, total_faults: float,
                              possibly_detected: float = 0, test_type: str = "stuck-at") -> dict:
    if total_faults <= 0:
        return {"error": "Total faults must be > 0"}
    coverage       = (detected_faults / total_faults) * 100
    with_possible  = ((detected_faults + possibly_detected) / total_faults) * 100
    undetected     = total_faults - detected_faults - possibly_detected
    undetected_pct = (undetected / total_faults) * 100
    if coverage >= 99:   grade, status = "A+", "Excellent — production ready"
    elif coverage >= 97: grade, status = "A",  "Very good — meets most tapeout requirements"
    elif coverage >= 95: grade, status = "B",  "Acceptable — marginal for high-reliability designs"
    elif coverage >= 90: grade, status = "C",  "Below target — investigate untestable faults"
    else:                grade, status = "F",  "Critical — significant DFT rework required"
    return {
        "test_type": test_type,
        "detected_faults": int(detected_faults), "total_faults": int(total_faults),
        "possibly_detected": int(possibly_detected), "undetected_faults": int(undetected),
        "fault_coverage_pct": round(coverage, 3),
        "coverage_with_pd_pct": round(with_possible, 3),
        "undetected_pct": round(undetected_pct, 3),
        "grade": grade, "status": status,
        "recommendation": (
            "Coverage meets industry standard." if coverage >= 97
            else "Review ATPG abort list, check sequential depth, add test points or increase effort."
        )
    }


def generate_debug_checklist(problem_type: str, design_context: str = "") -> dict:
    checklists = {
        "scan_chain_failure": {"title": "Scan Chain Failure Debug Checklist", "steps": [
            "1. IDENTIFY — Run Tessent `report_scan_chains` to list all chains and status",
            "2. ISOLATE — Use `set_scan_chain_enable` to disable half the chains (binary search)",
            "3. SHIFT TEST — Apply shift-only patterns to distinguish shift vs capture failure",
            "4. CHECK CONNECTIVITY — Verify scan-in/scan-out pin connections on ATE",
            "5. CLOCK CHECK — Confirm scan clock skew, duty cycle, voltage meets spec",
            "6. POWER DOMAINS — Check if failing chains cross power domain boundaries",
            "7. X-SOURCE HUNT — Run `report_logic_depth` to find X-propagation sources",
            "8. COMPARE SIM vs SI — Cross-check simulation waveforms against silicon",
            "9. TEMPERATURE/VOLTAGE — Re-run at nominal PVT corner if marginal",
            "10. EDT CHECK — If using compression, verify decompressor/compressor logic intact"
        ]},
        "low_fault_coverage": {"title": "Low Fault Coverage Debug Checklist", "steps": [
            "1. RUN FAULT REPORT — `report_faults -status undetected` to categorise untestable faults",
            "2. CHECK ATPG MODE — Confirm correct fault model is targeted",
            "3. REVIEW ABORTS — Check `report_abort_limit` — increase if needed",
            "4. SEQUENTIAL DEPTH — Deep sequential logic blocks ATPG; add test points",
            "5. CLOCK DOMAIN — Verify all clock domains correctly constrained in Tessent",
            "6. TIED SIGNALS — Identify tied-off logic creating ATPG-unreachable cones",
            "7. BLACK BOXES — Ensure all black boxes have proper models or wrappers",
            "8. COMPACTION — Run `set_pattern_compaction` to remove redundant patterns",
            "9. INCREMENTAL ATPG — Run targeted ATPG on remaining undetected fault list",
            "10. DFT RULES — Re-run DRC to catch any new violations after ECOs"
        ]},
        "mbist_failure": {"title": "MBIST Failure Debug Checklist", "steps": [
            "1. IDENTIFY MEMORY — Determine which instance is failing from BIST controller log",
            "2. ALGORITHM CHECK — Confirm March algorithm matches memory type (SRAM/ROM/CAM)",
            "3. TIMING — Verify BIST clock frequency within memory rated frequency",
            "4. POWER — Check VDD ramp and retention voltage for embedded memories",
            "5. REPAIR CHECK — If redundancy available run `analyse_memory_repair` in Tessent",
            "6. ADDRESS DECODE — Stuck address lines cause column/row failures — check pattern",
            "7. SENSE AMP — Weak bit-cells indicate sense amp marginal operation",
            "8. RETENTION TEST — Run write-then-delay-then-read for retention failures",
            "9. TEMPERATURE SWEEP — Re-run BIST at high/low temperature extremes",
            "10. COMPARE INSTANCES — Multiple failures → look for shared power/clock issue"
        ]},
        "jtag_failure": {"title": "JTAG / Boundary Scan Debug Checklist", "steps": [
            "1. TAP RESET — Apply 5x TCK with TMS=1 to force TAP to Test-Logic-Reset",
            "2. IDCODE — Read IDCODE register first; mismatch = wrong device or connection issue",
            "3. BYPASS TEST — Load BYPASS, shift 32 bits, verify 1-bit delay",
            "4. TCK FREQUENCY — Reduce TCK to 1MHz for initial bring-up debug",
            "5. CABLE/PROBE — Swap JTAG cable; check TDI/TDO/TMS/TCK pin mapping",
            "6. PULL-UPS — Verify TMS and TDI have correct pull-up/pull-down resistors",
            "7. BOUNDARY SCAN — Run SAMPLE/PRELOAD to capture I/O state",
            "8. EXTEST — Use EXTEST to drive and observe board-level interconnects",
            "9. DAISY CHAIN — Multiple devices: isolate each with BYPASS",
            "10. IR LENGTH — Confirm IR length matches BSDL file"
        ]},
        "silicon_bringup": {"title": "Silicon Bring-up Debug Checklist", "steps": [
            "1. POWER SEQUENCING — Verify all supply rails come up in correct order and within spec",
            "2. CLOCK — Confirm PLL lock and clock frequency at each domain",
            "3. RESET — Check reset de-assertion timing; many bring-up failures are reset related",
            "4. JTAG FIRST — Establish JTAG connectivity before anything else",
            "5. SCAN SHIFT — Run raw scan shift test at slow speed to verify chain connectivity",
            "6. STUCK-AT PATTERNS — Run stuck-at vectors first (most robust at bring-up)",
            "7. ATE CORRELATION — Compare bench results with ATE; timing margin differences common",
            "8. SUPPLY MARGINING — Run VDD sweep ±10% to find operating margin",
            "9. THERMAL — Monitor die temperature; thermal shutdown can mimic functional failures",
            "10. COMPARE REV — If re-spin, diff netlist against previous revision"
        ]},
        "atpg_abort": {"title": "ATPG Abort Debug Checklist", "steps": [
            "1. IDENTIFY ABORT TYPE — Check if aborts are backtrack limit, time limit, or conflict",
            "2. INCREASE LIMITS — `set_atpg_limit -backtrack 10000`",
            "3. FIND HARD FAULTS — `report_faults -status aborted`",
            "4. SEQUENTIAL DEPTH — Deep paths block ATPG; add scan flip-flops or test points",
            "5. ASYNCHRONOUS LOGIC — Async reset/set pins are common abort sources; constrain them",
            "6. TRISTATE BUSES — Add bus-hold models or constrain bus states in Tessent",
            "7. CLOCK GATING — Ensure all clock gates have correct enable constraints",
            "8. BIDIRECTIONAL I/O — Apply proper I/O models to prevent contention",
            "9. LOOPED LOGIC — Combinational loops cause ATPG to abort; break with test logic",
            "10. INCREMENTAL MODE — Use `-incremental` flag to target only newly aborted faults"
        ]}
    }
    result = checklists.get(problem_type)
    if not result:
        return {"error": f"Unknown problem type. Available: {list(checklists.keys())}"}
    if design_context:
        result["context"] = design_context
    result["problem_type"] = problem_type
    return result


def fetch_paper_summary(url: str) -> dict:
    try:
        if "arxiv.org" in url:
            arxiv_id = url.strip("/").split("/")[-1]
            req = urllib.request.Request(
                f"http://export.arxiv.org/api/query?id_list={arxiv_id}",
                headers={"User-Agent": "SiliconMind/2.0"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml_data = resp.read().decode("utf-8")
            ns    = {"atom": "http://www.w3.org/2005/Atom"}
            root  = ET.fromstring(xml_data)
            entry = root.find("atom:entry", ns)
            if entry is None:
                return {"error": "Paper not found"}
            return {
                "title":     entry.findtext("atom:title",   "", ns).strip(),
                "summary":   entry.findtext("atom:summary", "", ns).strip()[:800],
                "published": entry.findtext("atom:published","", ns)[:10],
                "url": url, "source": "ArXiv"
            }
        return {"url": url, "message": "Non-ArXiv URL — open link to read.", "source": "External"}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# SPRINT 1 — Netlist analysis (stolen + greatly improved from DFTAgent)
# ═══════════════════════════════════════════════════════════════════

def analyse_netlist(netlist_text: str, file_format: str = "auto") -> dict:
    """
    Parse Verilog, BENCH, or SPICE netlist text.
    Returns structured metrics + ATPG complexity estimate.
    DFTAgent only returns raw counts — we add AI-ready complexity analysis.
    """
    text = netlist_text.strip()
    fmt  = file_format.lower()

    # Auto-detect format
    if fmt == "auto":
        if re.search(r"\bmodule\b", text):           fmt = "verilog"
        elif re.search(r"^(INPUT|OUTPUT|AND|OR|NAND|NOR|NOT|XOR|DFF)\b", text, re.M): fmt = "bench"
        elif re.search(r"^[RCMLQDV]\w+\s", text, re.M): fmt = "spice"
        else:                                         fmt = "verilog"

    result = {"format": fmt, "status": "success"}

    # ── Verilog parser ──────────────────────────────────────────
    if fmt == "verilog":
        mod = re.search(r"module\s+(\w+)", text)
        result["module_name"] = mod.group(1) if mod else "unknown"

        # Port counting — handle multi-signal declarations
        def count_ports(keyword):
            matches = re.findall(rf"{keyword}\s+(?:\[\d+:\d+\]\s+)?([\w\s,]+);", text)
            return sum(len([s for s in m.split(",") if s.strip()]) for m in matches)

        result["input_ports"]  = count_ports("input")
        result["output_ports"] = count_ports("output")
        result["inout_ports"]  = count_ports("inout")
        result["wire_count"]   = count_ports("wire")
        result["reg_count"]    = count_ports("reg")

        # Gate/instance detection
        skip = {"input","output","inout","wire","reg","module","endmodule",
                "begin","end","always","assign","if","else","case","endcase"}
        instances = re.findall(r"^\s*(\w+)\s+(\w+)\s*\(", text, re.M)
        cells = [(t,n) for t,n in instances if t.lower() not in skip]
        result["cell_count"] = len(cells)

        # Gate type breakdown
        gate_types = {}
        for t, _ in cells:
            gate_types[t] = gate_types.get(t, 0) + 1
        result["gate_types"] = dict(sorted(gate_types.items(), key=lambda x: -x[1])[:10])

        # Sequential depth estimate
        ff_keywords = {"dff","ff","flop","reg","dffs","dffr","dffrs"}
        ff_count = sum(v for k,v in gate_types.items() if any(f in k.lower() for f in ff_keywords))
        result["estimated_ff_count"] = ff_count

    # ── BENCH parser (ISCAS format — used by Atalanta) ──────────
    elif fmt == "bench":
        result["module_name"] = "bench_circuit"
        result["input_ports"]  = len(re.findall(r"^INPUT\s*\(", text, re.M))
        result["output_ports"] = len(re.findall(r"^OUTPUT\s*\(", text, re.M))
        gate_matches = re.findall(r"=\s*(\w+)\s*\(", text)
        gate_types   = {}
        for g in gate_matches:
            gate_types[g.upper()] = gate_types.get(g.upper(), 0) + 1
        result["cell_count"] = len(gate_matches)
        result["gate_types"] = gate_types
        result["estimated_ff_count"] = gate_types.get("DFF", 0)

    # ── SPICE parser ─────────────────────────────────────────────
    elif fmt == "spice":
        result["module_name"] = "spice_circuit"
        components = re.findall(r"^([RCMLQDV])\w+", text, re.M)
        type_map   = {"R":"resistors","C":"capacitors","L":"inductors",
                      "M":"mosfets","Q":"bjts","D":"diodes","V":"vsources"}
        comp_types = {}
        for c in components:
            label = type_map.get(c, f"type_{c}")
            comp_types[label] = comp_types.get(label, 0) + 1
        result["cell_count"]    = len(components)
        result["component_breakdown"] = comp_types
        result["input_ports"]   = 0
        result["output_ports"]  = 0
        result["estimated_ff_count"] = 0

    # ── ATPG complexity estimate (added on top of DFTAgent) ──────
    cell_count = result.get("cell_count", 0)
    ff_count   = result.get("estimated_ff_count", 0)
    io_total   = result.get("input_ports", 0) + result.get("output_ports", 0)

    if cell_count < 100:
        complexity = "Low"
        atpg_time  = "< 1 second"
        expected_coverage = "95–100%"
    elif cell_count < 500:
        complexity = "Medium"
        atpg_time  = "5–30 seconds"
        expected_coverage = "90–98%"
    elif cell_count < 2000:
        complexity = "High"
        atpg_time  = "1–5 minutes"
        expected_coverage = "85–97%"
    else:
        complexity = "Very High"
        atpg_time  = "10+ minutes"
        expected_coverage = "80–95% without test points"

    seq_ratio = ff_count / max(cell_count, 1)
    if seq_ratio > 0.3:
        seq_note = "High sequential density — expect ATPG challenges; consider increasing scan insertion ratio."
    elif seq_ratio > 0.1:
        seq_note = "Moderate sequential logic — standard scan insertion should work well."
    else:
        seq_note = "Mostly combinational — ATPG should achieve high coverage quickly."

    result["atpg_analysis"] = {
        "complexity":         complexity,
        "estimated_atpg_time": atpg_time,
        "expected_coverage":  expected_coverage,
        "sequential_ratio":   round(seq_ratio, 3),
        "sequential_note":    seq_note,
        "scan_chain_estimate": max(1, cell_count // 200),
        "recommendation": (
            f"For {cell_count} gates with {ff_count} flip-flops: "
            f"use {'EDT compression' if cell_count > 500 else 'basic scan'}. "
            f"{seq_note}"
        )
    }
    return result


# ═══════════════════════════════════════════════════════════════════
# SPRINT 1 — Fault report parser (stolen + greatly improved from DFTAgent)
# ═══════════════════════════════════════════════════════════════════

def parse_fault_report(report_text: str, tool_hint: str = "auto") -> dict:
    """
    Parse fault reports from Tessent, Atalanta, or TetraMAX.
    DFTAgent extracts only coverage % with 4 regexes.
    We extract 12+ metrics and detect the tool automatically.
    """
    text  = report_text
    lower = text.lower()
    result = {"status": "success", "tool_detected": "unknown"}

    # ── Auto-detect tool ─────────────────────────────────────────
    if tool_hint == "auto":
        if "tessent" in lower or "fastscan" in lower:
            result["tool_detected"] = "tessent"
        elif "atalanta" in lower or "iscas" in lower:
            result["tool_detected"] = "atalanta"
        elif "tetramax" in lower or "synopsys" in lower:
            result["tool_detected"] = "tetramax"
    else:
        result["tool_detected"] = tool_hint

    # ── Coverage extraction — 10 patterns covering all tools ─────
    coverage = None
    coverage_patterns = [
        r"fault\s+coverage\s*[=:]\s*([0-9]+\.?[0-9]*)\s*%",
        r"coverage\s*[=:]\s*([0-9]+\.?[0-9]*)\s*%",
        r"([0-9]+\.?[0-9]*)\s*%\s*fault\s+coverage",
        r"([0-9]+\.?[0-9]*)\s*%\s*coverage",
        r"test\s+coverage\s*[=:]\s*([0-9]+\.?[0-9]*)",
        r"fc\s*[=:]\s*([0-9]+\.?[0-9]*)",
        r"detected\s+faults.*?([0-9]+\.?[0-9]*)\s*%",
        r"graded\s+faults.*?([0-9]+\.?[0-9]*)\s*%",
        r"total\s+coverage\s*[=:]\s*([0-9]+\.?[0-9]*)",
        r"([0-9]{2,3}\.[0-9]+)\s*%",
    ]
    for pat in coverage_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            coverage = float(m.group(1))
            break
    result["fault_coverage_pct"] = coverage

    # ── Extract additional metrics ───────────────────────────────
    def extract_int(patterns):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:    return int(m.group(1).replace(",",""))
                except: pass
        return None

    def extract_float(patterns):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:    return float(m.group(1))
                except: pass
        return None

    result["total_faults"] = extract_int([
        r"total\s+faults\s*[=:]\s*([0-9,]+)",
        r"fault\s+sites\s*[=:]\s*([0-9,]+)",
        r"([0-9,]+)\s+total\s+faults",
    ])
    result["detected_faults"] = extract_int([
        r"detected\s+faults\s*[=:]\s*([0-9,]+)",
        r"([0-9,]+)\s+detected",
    ])
    result["possibly_detected"] = extract_int([
        r"possibly\s+detected\s*[=:]\s*([0-9,]+)",
        r"atpg\s+detected\s*[=:]\s*([0-9,]+)",
    ])
    result["undetected_faults"] = extract_int([
        r"undetected\s*[=:]\s*([0-9,]+)",
        r"([0-9,]+)\s+undetected",
    ])
    result["aborted_faults"] = extract_int([
        r"aborted\s*[=:]\s*([0-9,]+)",
        r"abort\s+limit\s+reached\s*[=:]\s*([0-9,]+)",
        r"([0-9,]+)\s+aborted",
    ])
    result["untestable_faults"] = extract_int([
        r"untestable\s*[=:]\s*([0-9,]+)",
        r"redundant\s*[=:]\s*([0-9,]+)",
        r"not\s+detected\s*[=:]\s*([0-9,]+)",
    ])
    result["test_patterns"] = extract_int([
        r"test\s+patterns\s*[=:]\s*([0-9,]+)",
        r"patterns\s+generated\s*[=:]\s*([0-9,]+)",
        r"([0-9,]+)\s+patterns",
    ])
    result["cpu_time_sec"] = extract_float([
        r"cpu\s+time\s*[=:]\s*([0-9.]+)\s*s",
        r"elapsed\s+time\s*[=:]\s*([0-9.]+)",
        r"runtime\s*[=:]\s*([0-9.]+)",
    ])

    # ── Grade and recommendation ─────────────────────────────────
    if coverage is not None:
        if coverage >= 99:   grade, status = "A+", "Excellent — production ready"
        elif coverage >= 97: grade, status = "A",  "Meets most tapeout requirements"
        elif coverage >= 95: grade, status = "B",  "Marginal — review untestable faults"
        elif coverage >= 90: grade, status = "C",  "Below target — DFT rework needed"
        else:                grade, status = "F",  "Critical — major DFT issues present"
        result["grade"]  = grade
        result["status"] = status

        abort_count = result.get("aborted_faults") or 0
        untestedable = result.get("untestable_faults") or 0
        tips = []
        if abort_count > 0:
            tips.append(f"{abort_count} aborted faults — increase backtrack limit or add test points")
        if untestedable > 0:
            tips.append(f"{untestedable} untestable faults — check for tied-off logic or sequential loops")
        if coverage < 97:
            tips.append("Run `report_faults -status undetected` to analyse remaining faults")
        result["action_items"] = tips if tips else ["Coverage meets target — proceed to tapeout review"]
    else:
        result["status"]  = "parse_warning"
        result["message"] = "Could not extract fault coverage from this report. Check format."

    return result


# ═══════════════════════════════════════════════════════════════════
# SPRINT 3 — compare_fault_models  (original — no competitor has this)
# ═══════════════════════════════════════════════════════════════════

def compare_fault_models(design_type: str, process_node: str = "unknown",
                         quality_target: str = "general", constraints: str = "") -> dict:
    """
    Recommend fault model strategy based on design context.
    No existing DFT agent does this — unique to SiliconMind.
    """
    models = {
        "stuck_at": {
            "name": "Stuck-At (SAF)",
            "description": "Gates permanently stuck at logic 0 or 1",
            "coverage_typical": "95–99%",
            "pattern_count": "Low (1x baseline)",
            "atpg_time": "Fast",
            "detects": "Manufacturing defects: opens, shorts, oxide defects",
            "misses": "Timing-related defects, small delay faults",
            "when_to_use": "Always — mandatory baseline for all designs",
            "tessent_cmd": "set_fault_type stuck"
        },
        "transition_delay": {
            "name": "Transition Delay (TDF)",
            "description": "Slow-to-rise / slow-to-fall path timing faults",
            "coverage_typical": "90–97%",
            "pattern_count": "Medium (2–3x SAF)",
            "atpg_time": "Medium",
            "detects": "Timing defects, resistive opens, gate-oxide degradation",
            "misses": "Very small delays, some bridging faults",
            "when_to_use": "28nm and below, high-speed designs, automotive",
            "tessent_cmd": "set_fault_type transition"
        },
        "iddq": {
            "name": "IDDQ / Quiescent Current",
            "description": "Abnormal current draw in quiescent state",
            "coverage_typical": "Complementary to SAF",
            "pattern_count": "Low (subset of SAF patterns)",
            "atpg_time": "Fast (reuse SAF patterns)",
            "detects": "Bridging faults, oxide shorts, latch-up precursors",
            "misses": "Logical faults, open defects",
            "when_to_use": "65nm and above, analog-mixed signal, medical devices",
            "tessent_cmd": "set_fault_type iddq"
        },
        "cell_aware": {
            "name": "Cell-Aware (CAF)",
            "description": "Intra-cell defects based on transistor-level fault models",
            "coverage_typical": "Catches 3–8% additional defects vs SAF alone",
            "pattern_count": "High (3–5x SAF)",
            "atpg_time": "Slow",
            "detects": "Intra-cell bridges, opens invisible to SAF",
            "misses": "Inter-cell defects (covered by SAF)",
            "when_to_use": "7nm–28nm, automotive ASIL-D, medical, aerospace",
            "tessent_cmd": "set_fault_type cell_aware"
        },
        "path_delay": {
            "name": "Path Delay (PDF)",
            "description": "Actual path timing from input to output",
            "coverage_typical": "Subset of critical paths",
            "pattern_count": "Very High",
            "atpg_time": "Very Slow",
            "detects": "Critical timing paths, scan hold violations",
            "misses": "Non-critical paths (by design)",
            "when_to_use": "High-speed interfaces, memory controllers, CPUs",
            "tessent_cmd": "set_fault_type path_delay"
        },
        "bridging": {
            "name": "Bridging Faults",
            "description": "Unintended connections between adjacent nets",
            "coverage_typical": "Layout-dependent",
            "pattern_count": "High",
            "atpg_time": "Slow",
            "detects": "Metal shorts, via defects, layout-dependent defects",
            "misses": "Open defects",
            "when_to_use": "Dense layouts (7nm–14nm), post-layout extraction flows",
            "tessent_cmd": "set_fault_type bridging"
        }
    }

    # ── Recommendation engine ────────────────────────────────────
    recommended = ["stuck_at"]  # Always required
    priority_order = ["stuck_at"]
    rationale = ["Stuck-at is mandatory for all designs."]

    node_nm = 999
    node_match = re.search(r"(\d+)", process_node)
    if node_match:
        node_nm = int(node_match.group(1))

    # Transition delay
    if node_nm <= 65 or "automotive" in design_type.lower() or "high.speed" in design_type.lower():
        recommended.append("transition_delay")
        priority_order.append("transition_delay")
        rationale.append("Transition delay recommended for sub-65nm or automotive designs.")

    # Cell-aware
    qt = quality_target.lower()
    if "automotive" in qt or "iso26262" in qt or "asil" in qt or node_nm <= 28:
        recommended.append("cell_aware")
        priority_order.append("cell_aware")
        rationale.append("Cell-aware required for automotive ISO 26262 / ASIL-D or 28nm and below.")

    # IDDQ
    if "medical" in qt or "aerospace" in qt or node_nm >= 65:
        recommended.append("iddq")
        priority_order.append("iddq")
        rationale.append("IDDQ useful for 65nm+, medical, or aerospace for bridging detection.")

    # Path delay
    if "cpu" in design_type.lower() or "memory" in design_type.lower() or "high.speed" in design_type.lower():
        recommended.append("path_delay")
        priority_order.append("path_delay")
        rationale.append("Path delay for critical timing paths in CPUs / memory controllers.")

    # Bridging
    if node_nm <= 14:
        recommended.append("bridging")
        priority_order.append("bridging")
        rationale.append("Bridging faults important at 14nm and below due to dense metal layers.")

    run_order = priority_order  # already in recommended run order
    tessent_flow = "\n".join([
        f"# Step {i+1}: {models[m]['name']}\n{models[m]['tessent_cmd']}"
        for i, m in enumerate(run_order)
    ])

    return {
        "design_type":    design_type,
        "process_node":   process_node,
        "quality_target": quality_target,
        "recommended_models":   recommended,
        "run_order":             run_order,
        "rationale":             rationale,
        "model_details":         {k: models[k] for k in recommended},
        "tessent_flow_snippet":  tessent_flow,
        "not_recommended": [
            {"model": k, "reason": "Not required for this design context"}
            for k in models if k not in recommended
        ]
    }


# ═══════════════════════════════════════════════════════════════════
# SPRINT 3 — generate_tessent_script  (original — no competitor has this)
# ═══════════════════════════════════════════════════════════════════

def generate_tessent_script(task: str, design_name: str = "my_design", options: str = "") -> dict:
    """
    Generate a ready-to-run Tessent Shell TCL script.
    DFTAgent only works with Atalanta (open-source).
    This targets Tessent users — the real industry tool.
    """
    scripts = {
        "scan_insertion": {
            "title": "Scan Insertion",
            "description": "Full scan insertion flow for a synthesised netlist",
            "script": f"""# SiliconMind — Tessent Scan Insertion Script
# Design: {design_name}  |  Generated by SiliconMind
# ─────────────────────────────────────────────────

# 1. Setup
set_context dft -design_id rtl2

# 2. Read netlist
read_verilog {design_name}.v
read_verilog -library my_lib.v

# 3. Set top module
set_current_design {design_name}

# 4. Read constraints
read_sdc {design_name}.sdc

# 5. DFT rules check
set_system_mode analysis
report_dft_violations

# 6. Add scan chains
set_scan_configuration -style multiplexed_flip_flop
add_scan_chain_constraints -max_length 500

# 7. Insert scan
set_system_mode insertion
insert_test_logic

# 8. Connect scan chains  
connect_scan_chains -auto_master_clock

# 9. Verify
set_system_mode analysis
report_scan_chains
check_scan_rules

# 10. Write output
write_verilog {design_name}_scan.v
write_testbench -module {design_name}_tb -outdir ./tb
"""
        },
        "atpg_stuck_at": {
            "title": "ATPG Stuck-At Fault",
            "description": "Stuck-at ATPG pattern generation",
            "script": f"""# SiliconMind — Tessent ATPG Stuck-At Script
# Design: {design_name}  |  Generated by SiliconMind
# ─────────────────────────────────────────────────

# 1. Setup
set_context patterns -design_id rtl2

# 2. Read scan-inserted netlist
read_verilog {design_name}_scan.v
read_verilog -library my_lib.v
set_current_design {design_name}

# 3. Read constraints
read_pin_constraints {design_name}.pincon
read_sdc {design_name}.sdc

# 4. Set fault model
set_fault_type stuck

# 5. ATPG settings
set_atpg_limit -backtrack 5000
set_pattern_compaction on

# 6. Create patterns
set_system_mode analysis
create_patterns

# 7. Report
report_statistics
report_faults -status undetected
report_faults -status possibly_detected

# 8. Write patterns
write_patterns {design_name}_stuck.stil -format stil
write_patterns {design_name}_stuck.wgl  -format wgl
"""
        },
        "atpg_transition": {
            "title": "ATPG Transition Delay Fault",
            "description": "Transition delay fault ATPG",
            "script": f"""# SiliconMind — Tessent ATPG Transition Delay Script
# Design: {design_name}  |  Generated by SiliconMind
# ─────────────────────────────────────────────────

set_context patterns -design_id rtl2
read_verilog {design_name}_scan.v
read_verilog -library my_lib.v
set_current_design {design_name}

read_pin_constraints {design_name}.pincon
read_sdc {design_name}.sdc

# Transition delay fault model
set_fault_type transition

# Launch-on-shift or launch-on-capture
set_atpg_limit -backtrack 5000
set_drc_handling d1 -auto

set_system_mode analysis
create_patterns

report_statistics
report_faults -status undetected

write_patterns {design_name}_transition.stil -format stil
write_patterns {design_name}_transition.wgl  -format wgl
"""
        },
        "mbist_setup": {
            "title": "Tessent MemoryBIST Setup",
            "description": "MBIST configuration and insertion",
            "script": f"""# SiliconMind — Tessent MemoryBIST Script
# Design: {design_name}  |  Generated by SiliconMind
# ─────────────────────────────────────────────────

set_context memory_bist -design_id rtl2
read_verilog {design_name}.v
read_verilog -library my_lib.v
set_current_design {design_name}

# Identify memories
set_system_mode analysis
report_memory_summary

# Configure MBIST
set_bist_controller -name MBIST_CTRL \\
    -clock {design_name}_clk \\
    -reset {design_name}_rst_n \\
    -reset_sense active_low

# Algorithm selection
set_memory_algorithm -algorithm march_c_minus

# Insert MBIST logic
set_system_mode insertion
insert_test_logic

# Verify
report_bist_summary
report_memory_instances

# Write output
write_verilog {design_name}_mbist.v
"""
        },
        "edt_setup": {
            "title": "EDT Scan Compression Setup",
            "description": "Embedded Deterministic Test compression configuration",
            "script": f"""# SiliconMind — Tessent EDT Compression Script
# Design: {design_name}  |  Generated by SiliconMind
# ─────────────────────────────────────────────────

set_context dft -design_id rtl2
read_verilog {design_name}.v
read_verilog -library my_lib.v
set_current_design {design_name}

# Enable EDT compression
set_scan_compression_configuration \\
    -compressor_count 4 \\
    -decompressor_count 4 \\
    -compression_ratio 50

# Scan chain configuration
set_scan_configuration -max_length 500
add_scan_chain_constraints

# Insert EDT + scan
set_system_mode insertion
insert_test_logic

# Verify compression
set_system_mode analysis
report_scan_compression_statistics
report_scan_chains

write_verilog {design_name}_edt.v
"""
        },
        "fault_report": {
            "title": "Fault Coverage Report",
            "description": "Generate detailed fault coverage report",
            "script": f"""# SiliconMind — Tessent Fault Report Script
# Design: {design_name}  |  Generated by SiliconMind
# ─────────────────────────────────────────────────

set_context patterns -design_id rtl2
read_verilog {design_name}_scan.v
set_current_design {design_name}

# Load existing patterns
read_patterns {design_name}_stuck.stil

set_system_mode analysis

# Summary
report_statistics

# Detailed fault reports
report_faults -all                           > {design_name}_all_faults.rpt
report_faults -status detected              > {design_name}_detected.rpt
report_faults -status undetected            > {design_name}_undetected.rpt
report_faults -status possibly_detected     > {design_name}_possible.rpt
report_faults -status atpg_untestable       > {design_name}_untestable.rpt

# Abort analysis
report_abort_list                           > {design_name}_aborts.rpt

puts "Reports written to {design_name}_*.rpt"
"""
        },
        "drc_check": {
            "title": "DFT Design Rule Check",
            "description": "Run DFT DRC to find violations before scan insertion",
            "script": f"""# SiliconMind — Tessent DRC Check Script
# Design: {design_name}  |  Generated by SiliconMind
# ─────────────────────────────────────────────────

set_context dft -design_id rtl2
read_verilog {design_name}.v
read_verilog -library my_lib.v
set_current_design {design_name}
read_sdc {design_name}.sdc

set_system_mode analysis

# Run DRC
report_dft_violations

# Specific checks
check_scan_rules -verbose
report_clock_domain_crossings
report_asynchronous_set_reset
report_tristate_elements
report_feedback_loops

# Summary
report_dft_violations -summary > {design_name}_drc.rpt
puts "DRC report written to {design_name}_drc.rpt"
"""
        },
        "scan_diagnosis": {
            "title": "Scan Chain Diagnosis",
            "description": "Diagnose scan chain failures from failing patterns",
            "script": f"""# SiliconMind — Tessent Scan Diagnosis Script
# Design: {design_name}  |  Generated by SiliconMind
# ─────────────────────────────────────────────────

set_context diagnosis -design_id rtl2
read_verilog {design_name}_scan.v
read_verilog -library my_lib.v
set_current_design {design_name}

# Load failing patterns (from ATE)
read_patterns {design_name}_failing.stil
read_patterns -failing_cycles {design_name}_failing_log.txt

set_system_mode analysis

# Run diagnosis
diagnose_failures -num_candidates 5

# Reports
report_diagnosis_summary
report_diagnosis_candidates -all > {design_name}_diagnosis.rpt

# Chain-level analysis
report_scan_chain_health
report_suspect_cells -top 20

puts "Diagnosis complete. See {design_name}_diagnosis.rpt"
"""
        }
    }

    script_data = scripts.get(task)
    if not script_data:
        return {
            "error": f"Unknown task '{task}'.",
            "available_tasks": list(scripts.keys())
        }

    # Inject options as comments if provided
    script = script_data["script"]
    if options:
        script = f"# Options: {options}\n\n" + script

    return {
        "task":        task,
        "title":       script_data["title"],
        "description": script_data["description"],
        "design_name": design_name,
        "script":      script,
        "usage": f"Save as {design_name}_{task}.tcl and run: tessent -shell < {design_name}_{task}.tcl"
    }


# ═══════════════════════════════════════════════════════════════════
# TOOL DISPATCHER
# ═══════════════════════════════════════════════════════════════════

def execute_tool(tool_name: str, tool_input: dict) -> dict:
    dispatch = {
        "search_arxiv":             lambda i: search_arxiv(**i),
        "search_semantic_scholar":  lambda i: search_semantic_scholar(**i),
        "calculate_fault_coverage": lambda i: calculate_fault_coverage(**i),
        "generate_debug_checklist": lambda i: generate_debug_checklist(**i),
        "fetch_paper_summary":      lambda i: fetch_paper_summary(**i),
        # Sprint 1
        "analyse_netlist":          lambda i: analyse_netlist(**i),
        "parse_fault_report":       lambda i: parse_fault_report(**i),
        # Sprint 3
        "compare_fault_models":     lambda i: compare_fault_models(**i),
        "generate_tessent_script":  lambda i: generate_tessent_script(**i),
    }
    fn = dispatch.get(tool_name)
    if fn:
        return fn(tool_input)
    return {"error": f"Unknown tool: {tool_name}"}
