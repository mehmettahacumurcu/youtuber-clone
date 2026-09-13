"""Audio-grounded ASR-garble correction tool (full-segment editor).

The pipeline's Whisper transcripts contain occasional garbles (mis-heard names, broken/merged
words) that bleed into every downstream artifact (LLM training spans, RAG context, teacher pairs).
This tool surfaces each suspicious segment, plays its EXACT audio, and lets you edit the WHOLE
segment text to match what he actually said — so you can fix several errors in one segment at once
(e.g. a mis-heard word AND a wrongly-merged word), not just the one token that was flagged.

Output is per-segment corrected text (data/corrections/segment_overrides.json). At apply time these
diffs become corpus corrections (durable, globalised where the wrong form is a non-word; otherwise
context-scoped) and re-running `pipeline.clean` propagates them through corpus -> RAG -> dataset.
Raw transcripts are never touched here.

Prereq:  .venv\\Scripts\\python.exe -m tools.build_correction_queue   (writes data/corrections/queue.jsonl)
Run:     .venv\\Scripts\\python.exe -m ui.correct_app   (opens http://localhost:7861)
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import gradio as gr
import soundfile as sf

from pipeline.clean.runner import apply_corrections

CORR_DIR = Path("data/corrections")
QUEUE = CORR_DIR / "queue.jsonl"
DECISIONS = CORR_DIR / "decisions.jsonl"
OVERRIDES = CORR_DIR / "segment_overrides.json"
KEEP_SET = CORR_DIR / "keep_set.json"
PAD = 0.35  # seconds of audio padding either side of the segment


BROAD = CORR_DIR / "broad_queue.jsonl"


def _load_queue() -> list[dict]:
    rows = []
    for fp in (QUEUE, BROAD):  # original 44 first, then the broad sweep (high-confidence first)
        if fp.exists():
            rows += [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def _load_decisions() -> dict[str, dict]:
    """Keyed by 'video_id:seg_index:token'; last write wins so the tool is resumable.
    Tolerates the OLD schema (wrong/right) and the NEW schema (original/corrected)."""
    out: dict[str, dict] = {}
    if DECISIONS.exists():
        for l in DECISIONS.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                out[d["key"]] = d
    return out


def _slice_audio(path: str, start: float, end: float):
    info = sf.info(path)
    sr = info.samplerate
    a = max(0, int((start - PAD) * sr))
    b = min(info.frames, int((end + PAD) * sr))
    clip, sr = sf.read(path, start=a, stop=b, dtype="int16")
    if clip.ndim > 1:
        clip = clip[:, 0]
    return (sr, clip)


def _highlight(text: str, pos: int, token: str) -> str:
    if pos < 0:
        return html.escape(text)
    end = pos + len(token)
    return (html.escape(text[:pos])
            + f"<mark style='background:#ffd54f;font-weight:700'>{html.escape(text[pos:end])}</mark>"
            + html.escape(text[end:]))


def _corrected_from_decision(dec: dict, original: str) -> str:
    """Resolve the corrected text for a prior decision under either schema."""
    if dec is None:
        return original
    if "corrected" in dec:  # new schema
        return dec["corrected"]
    if dec.get("action") == "correct" and dec.get("wrong"):  # old schema -> compute
        return apply_corrections(original, {dec["wrong"]: dec.get("right", "")})
    return original  # old 'real'/'skip'


class App:
    def __init__(self) -> None:
        self.queue = _load_queue()
        self.decisions = _load_decisions()

    def key(self, item: dict) -> str:
        return f"{item['video_id']}:{item['seg_index']}:{item['token']}"

    def render(self, i: int):
        n = len(self.queue)
        if n == 0:
            return ("Queue is empty — run tools.build_correction_queue first.", None, "", "", "")
        i = max(0, min(i, n - 1))
        it = self.queue[i]
        original = it["text"]
        dec = self.decisions.get(self.key(it))
        if dec:
            prefill = _corrected_from_decision(dec, original)
        else:
            prefill = it.get("suggest_text") or original  # broad-sweep items pre-fill the suggested fix
        status = ""
        if dec:
            act = dec.get("action")
            if act in ("edit", "correct"):
                status = "✅ corrected" + (" (re-edit to change)" if prefill != original else "")
            elif act in ("ok", "real"):
                status = "👍 marked all-correct"
            elif act == "skip":
                status = "⏭ skipped"
        n_decided = len([d for d in self.decisions.values() if d.get("action") != "skip"])
        verify = "  ·  ⚠ likely his REAL word — confirm by ear" if it.get("verify") else ""
        header = (f"### {i+1} / {n}  ·  flagged: `{it['token']}`{verify}\n"
                  f"**{it.get('title','')}**  ·  `{it['video_id']}`  ·  logprob {it.get('avg_logprob')}"
                  f"  ·  {status}  ·  {n_decided}/{n} decided")
        body = (
            f"<div style='color:#888;font-size:0.9em'>…{html.escape(it.get('prev_text',''))}</div>"
            f"<div style='font-size:1.15em;margin:8px 0'>{_highlight(original, it.get('char_pos',-1), it['token'])}</div>"
            f"<div style='color:#888;font-size:0.9em'>{html.escape(it.get('next_text',''))}…</div>"
            "<div style='color:#aaa;font-size:0.85em;margin-top:6px'>↑ original (yellow = what I flagged) · "
            "edit the full text below to match the audio — fix every error you hear</div>"
        )
        if not dec and it.get("suggest_text") and it["suggest_text"] != original:
            body += ("<div style='color:#7fd87f;font-size:0.85em;margin-top:4px'>"
                     "✎ box is pre-filled with a suggested fix — just confirm it against the audio, or edit</div>")
        try:
            audio = _slice_audio(it["audio"], it["start"], it["end"])
        except Exception as e:  # noqa: BLE001
            audio = None
            body += f"<div style='color:#c00'>audio error: {e}</div>"
        return header, audio, body, prefill

    def _record(self, i: int, action: str, corrected: str):
        it = self.queue[i]
        rec = {"key": self.key(it), "action": action, "video_id": it["video_id"],
               "seg_index": it["seg_index"], "start": it.get("start"), "token": it["token"],
               "original": it["text"], "corrected": corrected.strip()}
        self.decisions[self.key(it)] = rec
        with DECISIONS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._export()

    def _export(self) -> None:
        overrides, keep = [], []
        for d in self.decisions.values():
            act = d.get("action")
            orig = d.get("original")
            corr = d.get("corrected")
            if act in ("edit", "correct"):
                # resolve corrected text under either schema
                if corr is None and d.get("wrong") and orig is not None:
                    corr = apply_corrections(orig, {d["wrong"]: d.get("right", "")})
                if orig is not None and corr is not None and corr.strip() and corr.strip() != orig.strip():
                    overrides.append({"video_id": d["video_id"], "seg_index": d["seg_index"],
                                      "start": d.get("start"), "token": d.get("token"),
                                      "original": orig, "corrected": corr.strip()})
            elif act in ("ok", "real"):
                keep.append(d.get("token"))
        OVERRIDES.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")
        KEEP_SET.write_text(json.dumps(sorted({k for k in keep if k}), ensure_ascii=False, indent=2),
                            encoding="utf-8")

    def save(self, i, corrected):
        it = self.queue[i]
        action = "edit" if corrected.strip() != it["text"].strip() else "ok"
        self._record(i, action, corrected)
        return self.go(i, +1)

    def ok(self, i, corrected):
        self._record(i, "ok", self.queue[i]["text"])
        return self.go(i, +1)

    def skip(self, i, corrected):
        self._record(i, "skip", corrected)
        return self.go(i, +1)

    def go(self, i, delta):
        ni = max(0, min(i + delta, len(self.queue) - 1))
        return (ni, *self.render(ni))


def build() -> gr.Blocks:
    app = App()
    with gr.Blocks(title="Speaker ASR correction") as demo:
        gr.Markdown("# 🎧 Audio-grounded transcript correction\n"
                    "Listen, then **edit the full segment text** to match what he said — fix every error "
                    "in the segment, not just the highlighted one. Decisions autosave to `data/corrections/`.")
        idx = gr.State(0)
        header = gr.Markdown()
        audio = gr.Audio(label="segment audio", autoplay=True)
        body = gr.HTML()
        text = gr.Textbox(label="corrected segment text (edit freely, then Save)", lines=3)
        with gr.Row():
            b_save = gr.Button("✓ Save fix", variant="primary")
            b_ok = gr.Button("👍 All correct")
            b_skip = gr.Button("⏭ Skip")
        with gr.Row():
            b_prev = gr.Button("◀ Prev")
            b_next = gr.Button("Next ▶")

        outs = [idx, header, audio, body, text]
        b_save.click(app.save, [idx, text], outs)
        b_ok.click(app.ok, [idx, text], outs)
        b_skip.click(app.skip, [idx, text], outs)
        b_prev.click(lambda i: app.go(i, -1), [idx], outs)
        b_next.click(lambda i: app.go(i, +1), [idx], outs)
        demo.load(lambda: app.go(0, 0), None, outs)
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
