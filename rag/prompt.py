"""Gate (ground/refuse/flag), grounded-prompt assembly, and citation formatting.

TWO prompt eras live here:
- Legacy `build_prompt` — for the BASE completion models (raw context + question, no markup).
- The CANONICAL chat templates (SYS + `build_grounded_user`) — the single source of truth shared
  by training (v4 grounded rows), ui/speaker_studio.py, and rag/ask_grounded.py. Training format
  MUST equal inference format, so nothing may build its own variant of these strings.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rag.types import EvidenceSpan, Hit

REFUSAL_TR = "Bu konuya pek değinmemişim, elimde bununla ilgili bir şey yok."

# ---------------------------------------------------------------------------- canonical templates
# SYS is byte-identical to the system prompt of every voice-SFT training row (v3 and v4) and the
# exported Modelfile SYSTEM. Do not edit without retraining.
SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")

# Caps shared by inference AND the v4 grounded training rows (a 14B on an 8GB card is slow on long
# prompts; training must see the same span shape the model gets at inference).
MAX_SPANS = 3
SPAN_CHAR_CAP = 800

_GROUNDED_HEADER = (
    "Aşağıda bu konu hakkında DAHA ÖNCE KENDİ söylediklerin var. Bunlara dayanarak, "
    "öğrendiğin anlatım üslubunu koruyarak soruyu cevapla. Uydurma; bu sözlerdeki "
    "görüşü kendi ağzından akıcı şekilde anlat.")


def build_grounded_user(question: str, spans: list[str], max_spans: int = MAX_SPANS,
                        span_char_cap: int = SPAN_CHAR_CAP) -> str:
    """The ONE grounded user-turn format (system stays plain SYS in both modes).

    Empty spans -> bare question (the closed-book trained format)."""
    spans = [s.strip() for s in (spans or []) if s and s.strip()]
    if not spans:
        return question.strip()
    ctx = "\n".join(f"- {s[:span_char_cap]}" for s in spans[:max_spans])
    return f"{_GROUNDED_HEADER}\n\n[Senin sözlerin]\n{ctx}\n\n[Soru] {question.strip()}"


@dataclass
class Decision:
    grounded: bool
    flagged: bool = False
    hits: list[Hit] = field(default_factory=list)


def gate(hits: list[Hit], threshold: float, ungrounded: bool) -> Decision:
    top = max((h.rerank_score for h in hits), default=float("-inf"))
    if top >= threshold:
        return Decision(grounded=True, flagged=False, hits=hits)
    if ungrounded:
        return Decision(grounded=True, flagged=True, hits=[])
    return Decision(grounded=False, flagged=False, hits=[])


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(h.chunk.text.strip() for h in hits)
    parts = [context, question.strip()] if context else [question.strip()]
    return "\n\n".join(parts).strip() + "\n"


def _mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def format_citations(hits: list[Hit]) -> str:
    lines, seen = [], set()
    for h in hits:
        c = h.chunk
        key = (c.video_id, int(c.start))
        if key in seen:
            continue
        seen.add(key)
        url = f"https://www.youtube.com/watch?v={c.video_id}&t={int(c.start)}s"
        lines.append(f'  Kaynak: "{c.title}" @ {_mmss(c.start)}  {url}')
    return "\n".join(lines)


def format_evidence_citations(spans: Sequence[EvidenceSpan]) -> str:
    lines: list[str] = []
    seen: set[tuple[str, int]] = set()
    for span in spans:
        key = (span.video_id, int(span.start))
        if key in seen:
            continue
        seen.add(key)
        url = f"https://www.youtube.com/watch?v={span.video_id}&t={int(span.start)}s"
        lines.append(f'  Kaynak: "{span.title}" @ {_mmss(span.start)}  {url}')
    return "\n".join(lines)
