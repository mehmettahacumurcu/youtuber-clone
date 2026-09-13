"""Speaker Studio — one local UI to test the whole clone, with a local chat and speech interface.

  - pick the LLM and use the isolated authenticated voice worker,
  - chat in strict-grounded, grounded-with-fallback, or explicit free mode,
  - hit "read in his voice" to hear the final answer through XTTS followed by RVC.

The Studio process has no in-process Torch/XTTS/RVC dependency. The LLM is reached over Ollama
HTTP; RAG retrieval lives in the OTHER env (.venv) behind the persistent retrieve_server (port 7670),
and synthesis is delegated to the loopback voice worker.

  .venv-tts\\Scripts\\python.exe ui\\speaker_studio.py
then open http://127.0.0.1:7861
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

# ROOT = parent of ui/ — the same file runs from the dev repo AND the runnable folder.
if "YOUTUBER_INSTALL_ROOT" not in os.environ:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.paths import RuntimePaths
from runtime.settings import RuntimeSettings
from runtime.loopback_http import loopback_json
from ui.voice_client import VoiceClient

RUNTIME_SETTINGS = RuntimeSettings.from_environment()
RUNTIME_PATHS = RuntimePaths.from_settings(RUNTIME_SETTINGS)
ROOT = str(RUNTIME_PATHS.code_root)


OLLAMA = RUNTIME_SETTINGS.ollama_origin
LANG = "tr"

# The shipped four (2026-07-04 wrap): the v5 A/B-arm run + the run2 incumbent. The dev zoo
# (failed cosmos/llama/gemma runs, raw bases) stays in Ollama but out of the dropdown.
FINAL_LLMS = ["speaker-v5-a636", "speaker-v5-a742", "speaker-v5-b742", "speaker-qwen3-run2-ep2"]

# chat_runtime owns the lightweight schema-v2 client and canonical prompt construction.
sys.path.insert(0, ROOT)
from ui.answer_modes import (  # noqa: E402
    ANSWER_MODE_LABELS,
    DEFAULT_ANSWER_MODE_LABEL,
    answer_mode_from_label,
    uses_retrieval,
)
from ui.chat_runtime import (  # noqa: E402
    DEFAULT_ANSWER_TOKEN_CAP,
    MAX_ANSWER_TOKEN_CAP,
    MIN_ANSWER_TOKEN_CAP,
    chat_turn,
    ensure_answer_model_absent,
)

# ---------------------------------------------------------------- model discovery


def list_llms() -> list[str]:
    try:
        tags = loopback_json(f"{OLLAMA}/api/tags", timeout_s=5).get("models", [])
        names = [m["name"] for m in tags]
    except Exception:
        names = []
    # only the shipped four; if none are installed (fresh machine) fall back to everything
    finals = [n for n in names if any(n.startswith(p) for p in FINAL_LLMS)]
    names = finals or names

    def _rank(n: str) -> tuple:
        return (next((i for i, p in enumerate(FINAL_LLMS) if n.startswith(p)), len(FINAL_LLMS)), n)

    names.sort(key=_rank)
    return names or ["speaker-v5-a636:latest"]


@dataclass(frozen=True)
class TtsOption:
    """A stable UI selection for the fixed isolated voice service."""

    label: str = "Speaker voice service"


def discover_tts(
    paths: RuntimePaths | None = None, *, packaged: bool | None = None
) -> dict[str, TtsOption]:
    """The Studio never discovers or loads model files; the voice worker owns them."""
    return {"Speaker voice service": TtsOption()}


TTS_OPTS = discover_tts()

# ---------------------------------------------------------------- LLM + RAG

def retr_health() -> str:
    """Status pill for the RAG process supervised by the Studio service."""
    try:
        h = loopback_json(
            f"http://127.0.0.1:{RUNTIME_SETTINGS.rag_port}/health/ready",
            bearer=RUNTIME_SETTINGS.session_secret, timeout_s=3,
        )
        if h.get("status") == "ready":
            return '<span class="pill ok">RAG READY</span>'
        return '<span class="pill warm">RAG WARMING…</span>'
    except Exception:
        return '<span class="pill err">RAG UNAVAILABLE</span>'


def _mode_value(label: str) -> str:
    return {
        "Clean transcripts": "clean",
        "Clean transcripts + card hints": "clean_with_card_hints",
    }[label]


def chat(
    message,
    history,
    model,
    answer_mode_label,
    retrieval_mode,
    answer_token_cap=DEFAULT_ANSWER_TOKEN_CAP,
):
    return chat_turn(
        message,
        history,
        model,
        answer_mode_from_label(answer_mode_label),
        _mode_value(retrieval_mode),
        answer_token_cap=answer_token_cap,
    )


def retrieval_mode_update(answer_mode_label):
    mode = answer_mode_from_label(answer_mode_label)
    return gr.update(interactive=uses_retrieval(mode))


def refresh_models():
    global TTS_OPTS
    TTS_OPTS = discover_tts()
    llms = list_llms()
    return (gr.update(choices=llms, value=llms[0] if llms else None),
            gr.update(choices=list(TTS_OPTS), value=(list(TTS_OPTS) or [None])[0]))


# ---------------------------------------------------------------- voice service


def _msg_text(c) -> str:
    """Gradio may pass message content as a string, a list of parts, or part-dicts — flatten it."""
    if isinstance(c, str):
        return c
    if isinstance(c, (list, tuple)):
        return " ".join(_msg_text(x) for x in c)
    if isinstance(c, dict):
        return _msg_text(c.get("text") or c.get("content") or "")
    return str(c or "")


def _voice_client() -> VoiceClient:
    return VoiceClient(
        f"http://127.0.0.1:{RUNTIME_SETTINGS.voice_port}",
        RUNTIME_SETTINGS.session_secret or "",
    )


def speak(text, label: str, answer_model: str | None = None):
    text = text.strip() if isinstance(text, str) else ""
    if not text:
        return None
    if label not in TTS_OPTS:
        return None
    if RUNTIME_SETTINGS.low_vram_voice:
        if not isinstance(answer_model, str) or not answer_model.strip():
            raise ValueError("low-VRAM voice requires a selected answer model")
        ensure_answer_model_absent(answer_model)
    artifact = _voice_client().synthesize(text, uuid.uuid4().hex)
    waveform, sample_rate = sf.read(BytesIO(artifact.wav_bytes), dtype="float32", always_2d=False)
    if sample_rate != artifact.sample_rate:
        raise ValueError("voice worker returned an unexpected sample rate")
    return sample_rate, np.asarray(waveform, dtype=np.float32)


def read_last(history, label, answer_model):
    for m in reversed(history or []):
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            return speak(m["content"], label, answer_model)
    return None


# ---------------------------------------------------------------- UI (Kick style)

KICK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');

:root, .dark {
  --kick-green: #53fc18;
  --kick-green-dim: #3fd90e;
  --bg: #0b0e0f;
  --surface: #191b1f;
  --surface2: #24272c;
  --border: #2f353b;
  --text: #efeff1;
  --muted: #b0b6bd;
}
body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 28px !important; }
.block, .form, .panel, .accordion {
  background: var(--surface) !important;
  border: 1px solid var(--surface2) !important;
  border-radius: 10px !important;
}
label, .label-wrap span, .block-title, span[data-testid="block-info"] {
  color: var(--muted) !important; font-weight: 600 !important;
  text-transform: uppercase; font-size: 11px !important; letter-spacing: .06em;
}
/* Keep native checkbox/radio states visible against the dark theme. */
input:not([type="checkbox"]):not([type="radio"]), textarea, select {
  background: #0f1214 !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 8px !important;
}
input:focus, textarea:focus { border-color: var(--kick-green) !important; }
button.primary, .primary {
  background: var(--kick-green) !important; color: #000 !important;
  font-weight: 800 !important; border: none !important; border-radius: 8px !important;
  text-transform: uppercase; letter-spacing: .03em;
}
button.primary:hover, .primary:hover { background: var(--kick-green-dim) !important; }
button.secondary, .secondary {
  background: var(--surface2) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 8px !important; font-weight: 600 !important;
}
input[type="checkbox"], input[type="range"] { accent-color: var(--kick-green) !important; }
#chat { background: #0f1214 !important; border: 1px solid var(--surface2) !important; }
#chat .message, #chat .bot, #chat .user { border-radius: 8px !important; }
#chat .bot  { background: var(--surface) !important; border-left: 3px solid var(--kick-green) !important; }
#chat .user { background: var(--surface2) !important; }
#brand { background: transparent !important; border: none !important; }
#brand .wordmark {
  color: var(--kick-green); font-weight: 900; font-size: 30px; font-style: italic;
  letter-spacing: .02em; text-shadow: 0 0 18px rgba(83,252,24,.35);
}
#brand .studio { color: var(--text); font-weight: 800; font-size: 30px; margin-left: 10px; }
#brand .sub { color: var(--muted); font-size: 12px; margin-left: 14px; letter-spacing: .12em; }
.pill {
  display: inline-block; padding: 3px 12px; border-radius: 999px;
  font-size: 11px; font-weight: 800; letter-spacing: .06em;
}
.pill.ok   { background: rgba(83,252,24,.14); color: var(--kick-green); border: 1px solid var(--kick-green); }
.pill.warm { background: rgba(255,190,20,.12); color: #ffbe14; border: 1px solid #ffbe14; }
.pill.err  { background: rgba(255,60,60,.12);  color: #ff5c5c; border: 1px solid #ff5c5c; }
footer { display: none !important; }
.meta {
  color: var(--muted); font-size: 12px; padding: 2px 6px; letter-spacing: .03em;
}
.meta b { color: var(--kick-green); font-weight: 800; }
.fallback-warning {
  color: #ffbe14; background: rgba(255,190,20,.10); border: 1px solid #ffbe14;
  border-radius: 8px; font-size: 12px; font-weight: 700; padding: 8px 10px; margin-bottom: 5px;
}
.disclaimer {
  color: #ffbe14; font-size: 11px; padding: 8px 4px 0 4px; letter-spacing: .04em;
  border-top: 1px solid var(--surface2); margin-top: 10px;
}
"""

_FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"

def build_app() -> gr.Blocks:
    """Construct the UI only after the authenticated service has started RAG."""
    global TTS_OPTS
    TTS_OPTS = discover_tts()
    llms = list_llms()
    with gr.Blocks(title="YOUTUBER studio", css=KICK_CSS, js=_FORCE_DARK_JS,
                   theme=gr.themes.Base()) as app:
        with gr.Row():
            gr.HTML('<div><span class="wordmark">YOUTUBER</span><span class="studio">STUDIO</span>'
                    '<span class="sub">LLM + RAG + TTS TEST BENCH</span></div>', elem_id="brand")
            health = gr.HTML('<span class="pill warm">RAG …</span>')
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(height=480, type="messages", elem_id="chat", label="chat")
                with gr.Row():
                    msg = gr.Textbox(placeholder="Soru sor…", scale=6, container=False)
                    send = gr.Button("Gönder", variant="primary", scale=1)
                meta = gr.HTML()
                with gr.Row():
                    readb = gr.Button("Sesiyle oku", variant="primary")
                    clear = gr.Button("Temizle", variant="secondary")
                with gr.Accordion("Kaynak dökümü (RAG)", open=False):
                    ctx = gr.Markdown()
            with gr.Column(scale=3):
                llm = gr.Dropdown(llms, value=llms[0] if llms else None, label="LLM (Ollama)")
                tts = gr.Dropdown(list(TTS_OPTS), value=(list(TTS_OPTS) or [None])[0], label="Voice service")
                refresh = gr.Button("Modelleri yenile", variant="secondary")
                answer_mode = gr.Dropdown(list(ANSWER_MODE_LABELS), value=DEFAULT_ANSWER_MODE_LABEL, label="Answer mode")
                retrieval_mode = gr.Dropdown(
                    ["Clean transcripts", "Clean transcripts + card hints"],
                    value="Clean transcripts + card hints", label="Retrieval mode",
                )
                answer_token_cap = gr.Slider(
                    minimum=MIN_ANSWER_TOKEN_CAP, maximum=MAX_ANSWER_TOKEN_CAP,
                    value=DEFAULT_ANSWER_TOKEN_CAP, step=64, precision=0, label="Answer token cap",
                )
                checkb = gr.Button("RAG durumunu kontrol et", variant="secondary")
                audio = gr.Audio(label="his voice", autoplay=True)
                gr.HTML('<div class="disclaimer">Bu bir yapay zekâ klonudur; gerçek YouTuber değildir.</div>')

        send.click(chat, [msg, chatbot, llm, answer_mode, retrieval_mode, answer_token_cap], [chatbot, ctx, meta]).then(lambda: "", None, msg)
        msg.submit(chat, [msg, chatbot, llm, answer_mode, retrieval_mode, answer_token_cap], [chatbot, ctx, meta]).then(lambda: "", None, msg)
        answer_mode.change(retrieval_mode_update, answer_mode, retrieval_mode)
        readb.click(read_last, [chatbot, tts, llm], audio)
        clear.click(lambda: ([], "", ""), None, [chatbot, ctx, meta])
        refresh.click(refresh_models, None, [llm, tts])
        checkb.click(retr_health, None, health)
        app.load(retr_health, None, health)
    return app


def launch_developer() -> None:
    """Explicit research-only entry point; packaged builds use ``python -m ui``."""
    if RUNTIME_SETTINGS.packaged:
        raise RuntimeError("packaged Studio must be run with python -m ui")
    auth_env = os.environ.get("SPEAKER_AUTH", "")
    auth = tuple(auth_env.split(":", 1)) if ":" in auth_env else None
    no_browser = os.environ.get("YOUTUBER_NO_BROWSER") == "1"
    build_app().launch(
        server_name="127.0.0.1", server_port=RUNTIME_SETTINGS.studio_port,
        share=False if no_browser else os.environ.get("SPEAKER_SHARE") == "1", auth=auth,
        inbrowser=False,
    )


if __name__ == "__main__":
    launch_developer()
