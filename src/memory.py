"""
SiliconMind Memory — Persistent Chat History
─────────────────────────────────────────────
Saves all conversations to a local SQLite database.
Survives app restarts, session resets, and redeploys.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "./siliconmind_memory.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            role      TEXT,
            content   TEXT,
            agent     TEXT,
            tools     TEXT,
            session   TEXT
        )
    """)
    conn.commit()
    return conn


def save_message(role: str, content: str, agent: str = "",
                 tools: list = None, session: str = "default"):
    """Save a single message to the database."""
    conn = get_conn()
    conn.execute(
        "INSERT INTO conversations (timestamp, role, content, agent, tools, session) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            role,
            content[:2000],
            agent,
            ",".join(tools or []),
            session
        )
    )
    conn.commit()
    conn.close()


def load_history(session: str = "default", limit: int = 50) -> list:
    """Load recent messages for a session."""
    try:
        conn = get_conn()
        rows = conn.execute(
            "SELECT role, content, agent, tools, timestamp FROM conversations "
            "WHERE session = ? ORDER BY id DESC LIMIT ?",
            (session, limit)
        ).fetchall()
        conn.close()
        # Reverse so oldest is first
        return [
            {
                "role":      r[0],
                "content":   r[1],
                "agent":     r[2],
                "tools_used": r[3].split(",") if r[3] else [],
                "timestamp": r[4]
            }
            for r in reversed(rows)
        ]
    except Exception:
        return []


def get_sessions() -> list:
    """Get all unique session names."""
    try:
        conn = get_conn()
        rows = conn.execute(
            "SELECT DISTINCT session, MAX(timestamp) as last_used "
            "FROM conversations GROUP BY session ORDER BY last_used DESC"
        ).fetchall()
        conn.close()
        return [{"session": r[0], "last_used": r[1][:10]} for r in rows]
    except Exception:
        return []


def delete_session(session: str):
    """Delete all messages in a session."""
    conn = get_conn()
    conn.execute("DELETE FROM conversations WHERE session = ?", (session,))
    conn.commit()
    conn.close()
    