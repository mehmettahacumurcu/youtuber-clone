"""Persistence layer for the Gradio review app.

- Reference pool decisions  -> data/reference/pool/active.json
- Borderline segment votes  -> data/decisions.sqlite

This module is intentionally small and pure-Python so it can be unit-tested
without spinning up Gradio.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import REPO_ROOT, Settings


# -------- reference pool active set --------

def active_pool_path(settings: Settings) -> Path:
    return REPO_ROOT / "data" / "reference" / "pool" / "active.json"


def load_active_pool(settings: Settings) -> dict[str, Any]:
    p = active_pool_path(settings)
    if not p.exists():
        return {"active_indices": [], "updated_at_utc": None}
    return json.loads(p.read_text(encoding="utf-8"))


def save_active_pool(settings: Settings, active_indices: list[int]) -> Path:
    p = active_pool_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active_indices": sorted(set(int(i) for i in active_indices)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# -------- decisions sqlite --------

def _open_db(settings: Settings) -> sqlite3.Connection:
    db_path = settings.paths.decisions_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            start REAL NOT NULL,
            end REAL NOT NULL,
            similarity REAL,
            speaker TEXT,
            decision TEXT NOT NULL,
            notes TEXT,
            decided_at_utc TEXT NOT NULL,
            UNIQUE(video_id, start, end)
        )
        """
    )
    conn.commit()
    return conn


def record_decision(
    settings: Settings,
    *,
    video_id: str,
    start: float,
    end: float,
    decision: str,  # 'keep' | 'drop' | 'mixed' | 'skip'
    similarity: float | None = None,
    speaker: str | None = None,
    notes: str | None = None,
) -> None:
    conn = _open_db(settings)
    try:
        conn.execute(
            """
            INSERT INTO decisions (video_id, start, end, similarity, speaker, decision, notes, decided_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id, start, end) DO UPDATE SET
                decision=excluded.decision,
                similarity=excluded.similarity,
                speaker=excluded.speaker,
                notes=excluded.notes,
                decided_at_utc=excluded.decided_at_utc
            """,
            (
                video_id,
                float(start),
                float(end),
                similarity,
                speaker,
                decision,
                notes,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_decision(
    settings: Settings, *, video_id: str, start: float, end: float
) -> str | None:
    conn = _open_db(settings)
    try:
        row = conn.execute(
            "SELECT decision FROM decisions WHERE video_id=? AND start=? AND end=?",
            (video_id, float(start), float(end)),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def decision_counts(settings: Settings) -> dict[str, int]:
    conn = _open_db(settings)
    try:
        rows = conn.execute(
            "SELECT decision, COUNT(*) FROM decisions GROUP BY decision"
        ).fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        conn.close()
