"""
SiliconMind Feedback Logger
────────────────────────────
Sprint 2: Thumbs up/down rating system (stolen from ORAssistant).
Stores ratings in a local SQLite DB — becomes your quality dataset over time.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "./siliconmind_feedback.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            question    TEXT,
            agent       TEXT,
            answer      TEXT,
            rating      INTEGER,   -- 1 = thumbs up, -1 = thumbs down
            tools_used  TEXT,
            confidence  TEXT
        )
    """)
    conn.commit()
    return conn


def log_feedback(question: str, agent: str, answer: str,
                 rating: int, tools_used: list = None, confidence: str = "") -> bool:
    """Log a thumbs up (1) or thumbs down (-1) for an answer."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO feedback (timestamp, question, agent, answer, rating, tools_used, confidence) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                datetime.utcnow().isoformat(),
                question[:500],
                agent,
                answer[:1000],
                rating,
                ",".join(tools_used or []),
                confidence
            )
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def get_stats() -> dict:
    """Return summary stats for the sidebar."""
    try:
        conn  = _get_conn()
        cur   = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) FROM feedback")
        total, positive = cur.fetchone()
        total    = total    or 0
        positive = positive or 0
        conn.close()
        pct = round((positive / total * 100), 1) if total > 0 else 0
        return {"total_ratings": total, "positive": positive,
                "negative": total - positive, "satisfaction_pct": pct}
    except Exception:
        return {"total_ratings": 0, "positive": 0, "negative": 0, "satisfaction_pct": 0}
