"""Gradio TTS-dataset review: play each clip, see + fix its transcript, drop bad ones.

Edits/drops persist to data/tts/.review_state.json as you go (safe to close + resume). The big
"APPLY" button writes the corrected transcripts to metadata.csv and deletes the dropped wavs.

  .venv-tts\\Scripts\\python.exe ui\\tts_review.py
then open the printed http://127.0.0.1:7860 URL.
"""
from __future__ import annotations

import json
import os

import gradio as gr

TTS = "E:/youtuber-clone/data/tts"
META = f"{TTS}/metadata.csv"
STATE = f"{TTS}/.review_state.json"


def load_rows():
    rows = []
    for ln in open(META, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            name, text = ln.split("|", 2)[:2]
            rows.append((name, text))
    return rows


ROWS = load_rows()
STATE_D = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {"edits": {}, "drops": []}


def _save():
    json.dump(STATE_D, open(STATE, "w", encoding="utf-8"), ensure_ascii=False)


def view(i):
    i = max(0, min(int(i), len(ROWS) - 1))
    name, orig = ROWS[i]
    text = STATE_D["edits"].get(name, orig)
    dropped = name in STATE_D["drops"]
    status = (f"### {i+1} / {len(ROWS)}  —  `{name}`  {'🚫 DROPPED' if dropped else ''}\n"
              f"edits: {len(STATE_D['edits'])} · drops: {len(STATE_D['drops'])}")
    return f"{TTS}/wavs/{name}.wav", text, status, i


def save_next(i, text):
    name, orig = ROWS[int(i)]
    if text.strip() and text.strip() != orig:
        STATE_D["edits"][name] = text.strip()
    _save()
    return view(int(i) + 1)


def drop(i):
    name = ROWS[int(i)][0]
    if name not in STATE_D["drops"]:
        STATE_D["drops"].append(name)
    _save()
    return view(int(i) + 1)


def undrop(i):
    name = ROWS[int(i)][0]
    if name in STATE_D["drops"]:
        STATE_D["drops"].remove(name)
    _save()
    return view(i)


def apply_changes():
    out = []
    for name, text in ROWS:
        if name in STATE_D["drops"]:
            continue
        t = STATE_D["edits"].get(name, text)
        out.append(f"{name}|{t}|{t}")
    open(META, "w", encoding="utf-8").write("\n".join(out) + "\n")
    for name in STATE_D["drops"]:
        p = f"{TTS}/wavs/{name}.wav"
        if os.path.exists(p):
            os.remove(p)
    return f"✅ APPLIED — kept {len(out)}, dropped {len(STATE_D['drops'])}, edited {len(STATE_D['edits'])}. metadata.csv rewritten. Now re-zip: python re-zip step."


with gr.Blocks(title="TTS dataset review") as app:
    gr.Markdown("## TTS dataset review — listen, fix the text if wrong, drop bad clips\n"
                "Audio autoplays. Edit the transcript box if the words are wrong, then **Save & Next**. "
                "**Drop** removes a bad clip (music/other-voice/garbled). Progress autosaves.")
    idx = gr.State(0)
    status = gr.Markdown()
    audio = gr.Audio(label="clip", autoplay=True)
    text = gr.Textbox(label="transcript (edit if the words don't match the audio)", lines=3)
    with gr.Row():
        prevb = gr.Button("◀ Prev")
        dropb = gr.Button("🚫 Drop", variant="stop")
        undropb = gr.Button("↩ Un-drop")
        savenext = gr.Button("💾 Save & Next ▶", variant="primary")
    nextb = gr.Button("Skip (no save) ▶")
    jump = gr.Number(label="jump to # (1-based)", value=1, precision=0)
    jumpb = gr.Button("Go")
    applyb = gr.Button("APPLY all edits/drops to metadata.csv", variant="primary")
    applymsg = gr.Markdown()

    outs = [audio, text, status, idx]
    prevb.click(lambda i: view(int(i) - 1), [idx], outs)
    nextb.click(lambda i: view(int(i) + 1), [idx], outs)
    savenext.click(save_next, [idx, text], outs)
    dropb.click(drop, [idx], outs)
    undropb.click(undrop, [idx], outs)
    jumpb.click(lambda j: view(int(j) - 1), [jump], outs)
    applyb.click(apply_changes, [], [applymsg])
    app.load(lambda: view(0), [], outs)

if __name__ == "__main__":
    app.launch()
