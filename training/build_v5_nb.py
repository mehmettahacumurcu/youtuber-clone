"""Generator for training/qwen_v5_train.ipynb — v5 production training run.

Production two-arm SFT at the safe low-LR decade (LR 1.5e-5, 15% Turkish replay).
Both arms use the identical recipe; the only variable is the base model.

  Arm A: huihui-ai/Huihui-Qwen3-14B-abliterated-v2  (abliterated)
  Arm B: Qwen/Qwen3-14B                              (stock)

Data: voice_sft_v4_train.jsonl (2,888 rows) + 15% Turkish replay.
max_steps=848 (~2 epochs mixed data), SAVE_EVERY=106 -> 8 checkpoints per arm.
20 probes total: 14 core (copied from probe) + 6 extracted from v4 val at build time.

Run:  .venv\\Scripts\\python.exe -m training.build_v5_nb
"""
from __future__ import annotations

import json
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.prompt import SYS as CANON_SYS  # noqa: E402

# ---- byte-identity guard: the notebook hardcodes SYS; it MUST equal rag.prompt.SYS
SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")
assert SYS == CANON_SYS, "SYS drifted from rag.prompt.SYS — fix before generating"

# ---- 14 probes copied verbatim from build_probe_nb.py for direct comparability
PROBES_14 = [
    {"kind": "opinion", "topic": "mutlak butlan",  "q": "Mutlak butlan meselesi hakkinda ne dusunuyorsun?",                       "kw": ["butlan"]},
    {"kind": "opinion", "topic": "dersimli kk",    "q": "Kemal Kilicdaroglu'nun Dersimli olmasi meselesini nasil yorumluyorsun?", "kw": ["dersim", "kilicdaroglu", "kemal"]},
    {"kind": "opinion", "topic": "almanci",         "q": "Almanci ne demek, gurbetciden farki ne?",                                "kw": ["almanci", "gurbetci"]},
    {"kind": "opinion", "topic": "12 eylul",        "q": "12 Eylul darbesi ve Kenan Evren hakkinda ne dusunuyorsun?",             "kw": ["eylul", "evren", "darbe"]},
    {"kind": "opinion", "topic": "ataturk",         "q": "Ataturk hakkinda ne dusunuyorsun?",                                     "kw": ["ataturk", "mustafa kemal"]},
    {"kind": "opinion", "topic": "pkk sureci",      "q": "PKK acilim sureci hakkinda ne dusunuyorsun?",                           "kw": ["pkk", "surec", "acilim"]},
    {"kind": "opinion", "topic": "ocalan",          "q": "Ocalan hakkinda ne dusunuyorsun?",                                      "kw": ["ocalan", "apo"]},
    {"kind": "opinion", "topic": "hitler",          "q": "Hitler ve Nazizm hakkinda ne dusunuyorsun?",                            "kw": ["hitler", "nazi"]},
    {"kind": "opinion", "topic": "askeri vesayet",  "q": "Turkiye'de askeri vesayet meselesi neydi?",                             "kw": ["vesayet", "ordu", "asker"]},
    {"kind": "opinion", "topic": "suriye",          "q": "Suriye politikamiz hakkinda ne dusunuyorsun?",                          "kw": ["suriye", "esad"]},
    {"kind": "voice",   "topic": "chp",             "q": "CHP hakkinda ne dusunuyorsun?",                                         "kw": ["chp"]},
    {"kind": "voice",   "topic": "milliyetcilik",   "q": "Turk milliyetciligi nedir sence?",                                      "kw": ["milliyet", "turk"]},
    {"kind": "fact",    "topic": "cumhuriyet yili", "q": "Turkiye Cumhuriyeti kac yilinda kuruldu?",                              "kw": ["1923"]},
    {"kind": "fact",    "topic": "obscure fact",    "q": "Sabetay Sevi kac yilinda oldu?",                                        "kw": []},
]

# ---- Extract 6 additional probes from v4 val at BUILD TIME (runs locally, not in Colab)
_VAL_PATH     = ROOT / "data" / "dataset" / "voice_sft_v4_val.jsonl"
_ABSTAIN_PATH = ROOT / "data" / "dataset" / "abstain_v4.jsonl"


def _extract_val_probes():
    val_rows     = [json.loads(l) for l in _VAL_PATH.read_text(encoding="utf-8").splitlines()     if l.strip()]
    abstain_rows = [json.loads(l) for l in _ABSTAIN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Whitespace-normalise for matching
    def _norm(s):
        return " ".join(s.strip().split())

    # Build the set of abstain bait questions from abstain_v4.jsonl
    abstain_qs: set[str] = set()
    for ar in abstain_rows:
        for m in ar["messages"]:
            if m["role"] == "user":
                abstain_qs.add(_norm(m["content"]))
                break

    grounded: list[dict] = []
    abstain:  list[dict] = []

    for row in val_rows:
        msgs   = row["messages"]
        user_q = next((m["content"] for m in msgs if m["role"] == "user"),      "")
        asst_a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        if "[Senin sözlerin]" in user_q:
            # Grounded row — prompt is the FULL user content (spans + [Soru] question embedded)
            soru  = user_q.split("[Soru]")[-1].strip() if "[Soru]" in user_q else user_q
            topic = soru[:40].rstrip("?").strip()
            grounded.append({"kind": "grounded", "topic": topic, "q": user_q, "kw": [], "ref": asst_a})
        elif _norm(user_q) in abstain_qs:
            topic = user_q[:40].rstrip("?").strip()
            abstain.append({"kind": "abstain", "topic": topic, "q": user_q, "kw": [], "ref": asst_a})

    g_pick = grounded[:3]
    a_pick = abstain[:3]

    if len(g_pick) < 3:
        print(f"BUILD WARNING: only {len(g_pick)} grounded val rows found (wanted 3); using all.", file=sys.stderr)
    if len(a_pick) < 3:
        print(f"BUILD WARNING: only {len(a_pick)} abstain val rows found (wanted 3); using all.", file=sys.stderr)

    print(f"\nExtracted {len(g_pick)} grounded + {len(a_pick)} abstain probes from val:")
    for p in g_pick:
        print(f"  grounded  [{p['topic'][:40]}]  prompt_len={len(p['q'])} chars")
    for p in a_pick:
        print(f"  abstain   [{p['topic'][:40]}]")

    return g_pick + a_pick


PROBES_EXTRACTED = _extract_val_probes()
PROBES = PROBES_14 + PROBES_EXTRACTED
assert len(PROBES) == 20, f"Expected 20 probes, got {len(PROBES)}"

# ---- Step math (computed here so build-time print matches what the notebook asserts)
_VOICE_N_EXPECTED = 2888
_REPLAY_FRAC      = 0.15
_REPLAY_N         = math.ceil(_VOICE_N_EXPECTED * _REPLAY_FRAC / (1.0 - _REPLAY_FRAC))  # 510
_MIXED_N          = _VOICE_N_EXPECTED + _REPLAY_N                                          # 3398
_EFF_BATCH        = 8    # BS=2, GA=4
_STEPS_PER_EPOCH  = _MIXED_N // _EFF_BATCH                                                 # 424
_MAX_STEPS        = _STEPS_PER_EPOCH * 2                                                    # 848
_SAVE_EVERY       = 106
_EP_EQUIV         = _MAX_STEPS * _EFF_BATCH / _MIXED_N
_CKPT_STEPS       = list(range(_SAVE_EVERY, _MAX_STEPS + 1, _SAVE_EVERY))
assert _MAX_STEPS % _SAVE_EVERY == 0,         "MAX_STEPS not divisible by SAVE_EVERY"
assert 1.5 <= _EP_EQUIV <= 2.5,               f"epoch equiv {_EP_EQUIV:.2f} out of [1.5, 2.5]"
assert len(_CKPT_STEPS) == 8,                 f"expected 8 checkpoints, got {len(_CKPT_STEPS)}"

print(f"\nStep math:")
print(f"  voice_n={_VOICE_N_EXPECTED}, replay_n={_REPLAY_N} ({_REPLAY_FRAC*100:.0f}%), mixed_n={_MIXED_N}")
print(f"  eff_batch={_EFF_BATCH}, steps_per_epoch={_STEPS_PER_EPOCH}, max_steps={_MAX_STEPS} (~{_EP_EQUIV:.2f} epochs)")
print(f"  SAVE_EVERY={_SAVE_EVERY} -> {len(_CKPT_STEPS)} checkpoints: {_CKPT_STEPS}")

# ============================================================
CELLS: list[dict] = []


def md(src: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src: str) -> None:
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})


# ============================================================ TITLE
md(r"""# Speaker v5 — production SFT: Arm A (abliterated) · Arm B (stock Qwen3-14B)

**Purpose:** the production style-transfer run after the collapse-onset probe established that
LR 1e-4 breaks Turkish morphology at 70–105 steps and 40 % replay slows but does not prevent it.
v5 trains in the untried low-LR decade with light replay.

| arm | base model | r / α | LR | max\_steps | save every | checkpoints |
|---|---|---|---|---|---|---|
| **A** | `huihui-ai/Huihui-Qwen3-14B-abliterated-v2` | 16/32 | 1.5e-5 | 848 | 106 | 8 (steps 106–848) |
| **B** | `Qwen/Qwen3-14B` | 16/32 | 1.5e-5 | 848 | 106 | 8 (steps 106–848) |

- **Data:** `voice_sft_v4_train.jsonl` (2,888 rows — v4 adds grounded RAFT + abstention rows) + **15 %**
  Turkish replay from `CohereForAI/aya_dataset` (≈ 510 rows → 3,398 mixed rows total).
- **Replay sys prompt:** neutral `REPLAY_SYS` — intentionally NOT the persona SYS.
- **Probes:** 20 total — 14 core opinion/voice/fact probes (same as probe notebook for comparability)
  + 6 extracted at build time from the v4 val set (3 grounded + 3 abstain held-out rows).
- **eval\_loss** is logged per epoch for continuity with earlier runs, but it is **blind to Turkish
  morphology collapse** — coherence is judged by READING the probe outputs, not by this number.
- Arm A and Arm B can be trained in separate Colab sessions; checkpoints live on Drive and the
  generation pass (Section 8) rediscovers them automatically.
- **Hardware:** H100 recommended (~2–2.5 h per arm). A100/L4 also work.
- **~18 checkpoints × 20 probes for the generation pass** — split sessions if needed.
- **Run in Colab web (browser), not the VS Code extension.**
""")

# ============================================================ 1. INSTALL
md("## 1. Install")
code(r"""%%capture
!pip install unsloth
!pip install -q gguf protobuf sentencepiece datasets""")
code(r"""import unsloth, trl, transformers, peft, datasets
print("unsloth", unsloth.__version__, "| trl", trl.__version__,
      "| transformers", transformers.__version__, "| peft", peft.__version__,
      "| datasets", datasets.__version__)""")

# ============================================================ 2. AUTH + DRIVE + GPU
md("## 2. Auth + Drive + GPU check (interactive, once)")
code(r"""import os, torch
assert torch.cuda.is_available(), "No GPU — pick H100/A100/L4 in Runtime > Change runtime type."
_name = torch.cuda.get_device_name(0)
_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print("GPU:", _name, f"{_vram:.0f} GB")
assert _vram >= 22, f"Only {_vram:.0f} GB — 14B-4bit training needs >=24 GB."
from huggingface_hub import login
login()  # paste a READ token — needed for model + aya_dataset download
from google.colab import drive
drive.mount("/content/drive")
OUTPUT_ROOT = "/content/drive/MyDrive/speaker_models"
RUN = "v5"   # tags every output; never clobbers probe1 / v3sweep files on Drive
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print("outputs ->", OUTPUT_ROOT, "| run tag:", RUN)""")

# ============================================================ 3. DATA UPLOAD
md(r"""## 3. Data — upload `voice_sft_v4_train.jsonl` + `voice_sft_v4_val.jsonl`

v4 train adds grounded (RAFT) and abstention rows on top of the v3 voice backbone.
Right-click → Upload in the Colab file panel, or uncomment the git-clone block below.""")
code(r"""import glob as _glob, json as _json

def _find(name):
    for d in [".", "/content", "/content/drive/MyDrive"]:
        p = os.path.join(d, name)
        if os.path.isfile(p): return p
    hits = _glob.glob(f"/content/**/{name}", recursive=True) or _glob.glob(f"**/{name}", recursive=True)
    return hits[0] if hits else None

# Uncomment to clone repo and pull files:
# import getpass; PAT = getpass.getpass("GitHub PAT: ")
# !git clone --depth 1 https://{PAT}@github.com/mehmettahacumurcu/youtuber-clone.git _repo
# TRAIN = "_repo/data/dataset/voice_sft_v4_train.jsonl"
# VAL   = "_repo/data/dataset/voice_sft_v4_val.jsonl"

TRAIN = _find("voice_sft_v4_train.jsonl")
VAL   = _find("voice_sft_v4_val.jsonl")
assert TRAIN and VAL, "voice_sft_v4_train.jsonl / voice_sft_v4_val.jsonl not found — upload them."
_rows = [_json.loads(l) for l in open(TRAIN, encoding="utf-8") if l.strip()]
assert len(_rows) >= 2800, f"only {len(_rows)} rows — expected >= 2800 (v4 full set); wrong file?"
print(f"train: {TRAIN} ({len(_rows)} rows) | val: {VAL}")""")

# ============================================================ 4. CONFIG + STEP MATH
md(r"""## 4. Config — constants, arm parameters, step math

Both arms share the same recipe; the only difference is the base model loaded in Sections 6 and 7.

**Note on eval\_loss:** `eval_strategy="epoch"` is set for continuity with earlier runs. However,
eval\_loss on the style-SFT val set does NOT detect Turkish morphology collapse — a collapsing model
can show stable or even improving eval\_loss while producing word-salad. Collapse is caught by
reading the probe outputs in Section 8. The eval\_loss numbers are included for comparison only.""")
code(r"""# ── Base models (one-line swap per arm) ──────────────────────────────────────────────────────────
BASE_A = "huihui-ai/Huihui-Qwen3-14B-abliterated-v2"   # Arm A: abliterated
BASE_B = "Qwen/Qwen3-14B"                               # Arm B: stock

MAX_SEQ_LEN  = 4096
TARGETS      = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # all 7 linear
LORA_DROPOUT = 0.05
SEED         = 7
BS, GA       = 2, 4        # eff-batch 8; drop to 1,8 if you OOM

# Shared recipe (both arms)
LORA_R      = 16
LORA_ALPHA  = 32           # alpha = 2r
LR          = 1.5e-5
MAX_STEPS   = 848          # ~2 epochs of mixed data
SAVE_EVERY  = 106          # 8 checkpoints per arm: 106, 212, 318, 424, 530, 636, 742, 848
REPLAY_FRAC = 0.15         # 15 % replay rows in the combined dataset

# Persona system prompt — byte-identical to the dataset system prompt and rag/prompt.py SYS.
SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")

# Neutral replay system prompt — intentionally NOT the persona SYS.
# Replay teaches Turkish morphology/grammar, NOT the character. This separation is critical.
REPLAY_SYS = "Sana bir soru sorulur; net ve dogru Turkce yanit ver."

print("BASE_A:", BASE_A)
print("BASE_B:", BASE_B)
print(f"Recipe: r={LORA_R}, alpha={LORA_ALPHA}, LR={LR}, max_steps={MAX_STEPS}, SAVE_EVERY={SAVE_EVERY}")
print(f"replay_frac={REPLAY_FRAC}")""")
code(r"""import math as _math

# ── Step math — verify before training ───────────────────────────────────────────────────────────
_voice_n         = sum(1 for _l in open(TRAIN, encoding="utf-8") if _l.strip())
_eff_b           = BS * GA   # 8
_replay_n_need   = _math.ceil(_voice_n * REPLAY_FRAC / (1.0 - REPLAY_FRAC))
_mixed_n         = _voice_n + _replay_n_need
_steps_per_epoch = _mixed_n // _eff_b
_ep_equiv        = MAX_STEPS * _eff_b / _mixed_n

print(f"voice_n={_voice_n}, replay_n={_replay_n_need} ({REPLAY_FRAC*100:.0f}%), mixed_n={_mixed_n}")
print(f"eff_batch={_eff_b}, steps_per_epoch={_steps_per_epoch}")
print(f"MAX_STEPS={MAX_STEPS} (~{_ep_equiv:.2f} epochs of mixed data)")
print(f"SAVE_EVERY={SAVE_EVERY} -> {MAX_STEPS // SAVE_EVERY} checkpoints at steps: {list(range(SAVE_EVERY, MAX_STEPS + 1, SAVE_EVERY))}")

assert MAX_STEPS % SAVE_EVERY == 0, \
    f"MAX_STEPS={MAX_STEPS} must be divisible by SAVE_EVERY={SAVE_EVERY} for complete checkpoint coverage"
assert 1.5 <= _ep_equiv <= 2.5, \
    f"epoch equiv {_ep_equiv:.2f} outside expected [1.5, 2.5] — check voice_n / eff_batch / MAX_STEPS"
assert MAX_STEPS // SAVE_EVERY == 8, \
    f"expected 8 checkpoints per arm, got {MAX_STEPS // SAVE_EVERY}"
print("Step math asserts passed.")""")

# ============================================================ 5. HELPERS + REPLAY
md(r"""## 5. Helpers — load_fresh_arm, make_peft, fmt_ds, StepSaver callback

`load_fresh_arm(base)` loads a CLEAN 4-bit model from the given base each time, preventing
optimizer / adapter state leaks between arms. Both arms are otherwise identical.""")
code(r"""import torch, time, gc
from datasets import load_dataset, Dataset
from transformers import TrainerCallback

def load_fresh_arm(base):
    # Load a clean 4-bit base model + thinking-free Qwen-2.5 chat template.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    model, tok = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    tok = get_chat_template(tok, chat_template="qwen-2.5", map_eos_token=True)
    tok.truncation_side = "left"   # overflow trims the FRONT (context/spans), never the answer+<|im_end|>
    return model, tok

def make_peft(model):
    # Attach r=16/alpha=32 LoRA and assert embeddings/lm_head stay frozen.
    from unsloth import FastLanguageModel
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=TARGETS, use_gradient_checkpointing="unsloth", random_state=SEED)
    for n_, p_ in model.named_parameters():
        if ("embed_tokens" in n_ or "lm_head" in n_) and p_.requires_grad:
            raise RuntimeError(f"embeddings trainable ({n_}) -- must be frozen")
    return model

def fmt_ds(raw_ds, tok):
    # Apply chat template to a dataset with a 'messages' column.
    def _fmt(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}
    return raw_ds.map(_fmt)


class StepSaver(TrainerCallback):
    # Saves LoRA adapter + tokenizer to Drive every `save_every` optimizer steps.
    def __init__(self, arm, save_every, tok, saved_dict):
        self.arm       = arm
        self.save_every = save_every
        self.tok       = tok
        self.saved     = saved_dict

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step > 0 and state.global_step % self.save_every == 0:
            p = f"{OUTPUT_ROOT}/qwen3_{RUN}_{self.arm}_step{state.global_step:04d}"
            model.save_pretrained(p)
            self.tok.save_pretrained(p)
            self.saved[state.global_step] = p
            print(f"  [ckpt] {self.arm} step{state.global_step:04d} -> {p}")""")

# ---- Replay construction cell (shared by both arms)
md(r"""### 5a. Replay construction (run once, shared by both arms)

Loads `CohereForAI/aya_dataset`, filters Turkish rows, samples ~15 % of the combined dataset.
Replay rows carry `REPLAY_SYS` — never the persona `SYS`. This is intentional and critical:
replay teaches Turkish morphology/grammar, not the character voice.""")
code(r"""import math as _math2, random, json as _json_r
from datasets import load_dataset as _lds

# ── Compute exact number of replay rows needed ────────────────────────────────────────────────────
_voice_n_r   = sum(1 for _l in open(TRAIN, encoding="utf-8") if _l.strip())
_replay_need = _math2.ceil(_voice_n_r * REPLAY_FRAC / (1.0 - REPLAY_FRAC))
print(f"Need {_replay_need} replay rows for REPLAY_FRAC={REPLAY_FRAC} ({_voice_n_r} voice rows)")

# ── Load Turkish subset of aya_dataset ───────────────────────────────────────────────────────────
print("\nLoading CohereForAI/aya_dataset ...")
_aya = _lds("CohereForAI/aya_dataset", split="train")
print(f"  columns: {_aya.column_names}")
_lang_col = None
for _c in _aya.column_names:
    if "lang" in _c.lower():
        _lang_col = _c; break
assert _lang_col is not None, f"No language column in aya_dataset — columns: {_aya.column_names}"
print(f"  language column: '{_lang_col}'")
_sample_langs = list(set(_aya.select(range(min(500, len(_aya))))[_lang_col]))
print(f"  sample language values: {sorted(_sample_langs)[:30]}")
_tur_ids  = {"Turkish", "tr", "tur", "Türkçe", "turkish"}
_aya_tr   = _aya.filter(lambda ex: ex[_lang_col] in _tur_ids)
print(f"  Turkish rows found: {len(_aya_tr)}")
assert len(_aya_tr) > 0, (
    f"No Turkish rows found (lang_col='{_lang_col}', tried ids={_tur_ids}). "
    "Check sample above and add the correct identifier to _tur_ids.")

# ── Sample replay rows ────────────────────────────────────────────────────────────────────────────
random.seed(SEED)
if len(_aya_tr) >= _replay_need:
    _replay_indices = random.sample(range(len(_aya_tr)), _replay_need)
    print(f"  sampled {_replay_need} replay rows WITHOUT replacement")
else:
    _rep_factor = _replay_need / len(_aya_tr)
    print(f"  WARNING: Turkish subset ({len(_aya_tr)} rows) < needed ({_replay_need}). "
          f"Sampling WITH REPLACEMENT ({_rep_factor:.2f}x).")
    _replay_indices = [random.choice(range(len(_aya_tr))) for _ in range(_replay_need)]

# ── Convert to chat format with NEUTRAL system prompt ────────────────────────────────────────────
_input_col  = "inputs"  if "inputs"  in _aya_tr.column_names else _aya_tr.column_names[1]
_target_col = "targets" if "targets" in _aya_tr.column_names else _aya_tr.column_names[2]
print(f"  aya input_col='{_input_col}', target_col='{_target_col}'")

_replay_rows = []
for _idx in _replay_indices:
    _ex = _aya_tr[_idx]
    _replay_rows.append({"messages": [
        {"role": "system",    "content": REPLAY_SYS},
        {"role": "user",      "content": str(_ex[_input_col])},
        {"role": "assistant", "content": str(_ex[_target_col])},
    ]})

# ── Invariant asserts ─────────────────────────────────────────────────────────────────────────────
for _rr in _replay_rows[:20]:
    assert _rr["messages"][0]["content"] != SYS, \
        "BUG: replay row carries the persona SYS — must use REPLAY_SYS"
print("  assert: replay rows carry REPLAY_SYS (not persona SYS) [OK]")

_combined_n = _voice_n_r + len(_replay_rows)
_real_frac  = len(_replay_rows) / _combined_n
assert abs(_real_frac - REPLAY_FRAC) <= 0.02, (
    f"Realized replay fraction {_real_frac:.3f} outside ±2% of REPLAY_FRAC={REPLAY_FRAC}. "
    f"voice={_voice_n_r}, replay={len(_replay_rows)}, combined={_combined_n}.")
print(f"  assert: realized replay fraction {_real_frac:.3f} within ±2% of {REPLAY_FRAC} [OK]")
print(f"\nReplay ready: {_voice_n_r} voice + {len(_replay_rows)} replay = {_combined_n} combined rows")""")

# ============================================================ 6. ARM A
md(r"""## 6. Arm A — `huihui-ai/Huihui-Qwen3-14B-abliterated-v2` (max_steps=848, save every 106 steps)

Abliterated variant: refusal behaviour has been removed, so it follows the persona system prompt
without content-filtering interference. Training is identical to Arm B; the A/B comparison isolates
base-model effects on voice fidelity and Turkish robustness.

**~2–2.5 h on H100.**""")
code(r"""from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# ── Build mixed dataset (voice + replay, shuffled) ───────────────────────────────────────────────
_voice_rows_A = [_json_r.loads(_l) for _l in open(TRAIN, encoding="utf-8") if _l.strip()]
_mixed_raw_A  = _voice_rows_A + _replay_rows
random.seed(SEED)
random.shuffle(_mixed_raw_A)
_mixed_ds_A = Dataset.from_list(_mixed_raw_A)
print(f"Arm A mixed dataset: {len(_mixed_ds_A)} rows "
      f"({len(_voice_rows_A)} voice + {len(_replay_rows)} replay), shuffled (seed={SEED})")
print(f"Steps per pass: {len(_mixed_ds_A) // (BS * GA)}")

model_A, tok_A = load_fresh_arm(BASE_A)
model_A        = make_peft(model_A)
train_ds_A     = fmt_ds(_mixed_ds_A,                                     tok_A)
val_ds_A       = fmt_ds(load_dataset("json", data_files=VAL, split="train"), tok_A)

saved_A = {}
saver_A = StepSaver("A", SAVE_EVERY, tok_A, saved_A)

args_A = SFTConfig(
    output_dir="/content/out_A", dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
    per_device_train_batch_size=BS, gradient_accumulation_steps=GA,
    max_steps=MAX_STEPS,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.05,
    weight_decay=0.01, optim="adamw_8bit", logging_steps=10,
    eval_strategy="epoch", per_device_eval_batch_size=1, prediction_loss_only=True,
    save_strategy="no", seed=SEED, report_to="none",
    fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported())
try:
    tr_A = SFTTrainer(model=model_A, tokenizer=tok_A, train_dataset=train_ds_A,
                      eval_dataset=val_ds_A, args=args_A, callbacks=[saver_A])
except TypeError:
    tr_A = SFTTrainer(model=model_A, processing_class=tok_A, train_dataset=train_ds_A,
                      eval_dataset=val_ds_A, args=args_A, callbacks=[saver_A])
tr_A = train_on_responses_only(tr_A, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

t0 = time.time()
tr_A.train()
# Safety save: defend against edge cases where on_step_end doesn't fire at the final step
if MAX_STEPS not in saved_A:
    _p = f"{OUTPUT_ROOT}/qwen3_{RUN}_A_step{MAX_STEPS:04d}"
    model_A.save_pretrained(_p); tok_A.save_pretrained(_p); saved_A[MAX_STEPS] = _p
    print(f"  [safety-save] A step{MAX_STEPS:04d} -> {_p}")
print(f"\nARM A DONE: {(time.time()-t0)/60:.1f} min | {len(saved_A)} checkpoints: {sorted(saved_A)}")
del tr_A, model_A, tok_A; gc.collect(); torch.cuda.empty_cache()""")

# ============================================================ 7. ARM B
md(r"""## 7. Arm B — `Qwen/Qwen3-14B` stock (max_steps=848, save every 106 steps)

Stock Qwen3-14B base — no abliteration. Recipe identical to Arm A. The blind A/B comparison
reveals whether abliteration meaningfully affects voice fidelity or Turkish robustness at this LR.

**~2–2.5 h on H100.** Can run in a separate Colab session if time is limited — checkpoints from
Arm A are already on Drive, and the generation pass (Section 8) discovers both arms automatically.
**Fresh session? Run Sections 1–5a first** — this cell needs `_replay_rows` from 5a. The replay
cell is deterministic (seed 7), so both arms get byte-identical training data even across sessions.""")
code(r"""from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# ── Build mixed dataset for Arm B (same data, same shuffle seed) ─────────────────────────────────
_voice_rows_B = [_json_r.loads(_l) for _l in open(TRAIN, encoding="utf-8") if _l.strip()]
_mixed_raw_B  = _voice_rows_B + _replay_rows
random.seed(SEED)
random.shuffle(_mixed_raw_B)
_mixed_ds_B = Dataset.from_list(_mixed_raw_B)
print(f"Arm B mixed dataset: {len(_mixed_ds_B)} rows "
      f"({len(_voice_rows_B)} voice + {len(_replay_rows)} replay), shuffled (seed={SEED})")
print(f"Steps per pass: {len(_mixed_ds_B) // (BS * GA)}")

model_B, tok_B = load_fresh_arm(BASE_B)
model_B        = make_peft(model_B)
train_ds_B     = fmt_ds(_mixed_ds_B,                                     tok_B)
val_ds_B       = fmt_ds(load_dataset("json", data_files=VAL, split="train"), tok_B)

saved_B = {}
saver_B = StepSaver("B", SAVE_EVERY, tok_B, saved_B)

args_B = SFTConfig(
    output_dir="/content/out_B", dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
    per_device_train_batch_size=BS, gradient_accumulation_steps=GA,
    max_steps=MAX_STEPS,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.05,
    weight_decay=0.01, optim="adamw_8bit", logging_steps=10,
    eval_strategy="epoch", per_device_eval_batch_size=1, prediction_loss_only=True,
    save_strategy="no", seed=SEED, report_to="none",
    fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported())
try:
    tr_B = SFTTrainer(model=model_B, tokenizer=tok_B, train_dataset=train_ds_B,
                      eval_dataset=val_ds_B, args=args_B, callbacks=[saver_B])
except TypeError:
    tr_B = SFTTrainer(model=model_B, processing_class=tok_B, train_dataset=train_ds_B,
                      eval_dataset=val_ds_B, args=args_B, callbacks=[saver_B])
tr_B = train_on_responses_only(tr_B, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

t0 = time.time()
tr_B.train()
if MAX_STEPS not in saved_B:
    _p = f"{OUTPUT_ROOT}/qwen3_{RUN}_B_step{MAX_STEPS:04d}"
    model_B.save_pretrained(_p); tok_B.save_pretrained(_p); saved_B[MAX_STEPS] = _p
    print(f"  [safety-save] B step{MAX_STEPS:04d} -> {_p}")
print(f"\nARM B DONE: {(time.time()-t0)/60:.1f} min | {len(saved_B)} checkpoints: {sorted(saved_B)}")
del tr_B, model_B, tok_B; gc.collect(); torch.cuda.empty_cache()""")

# ============================================================ 8. GENERATE PROBE OUTPUTS
md(r"""## 8. Generate probe outputs from every checkpoint

Re-runnable from a fresh runtime — rediscovers checkpoints from Drive. Probes BOTH base models with
no adapter (step-0 reference) before scoring their checkpoints.

**Checkpoint order:** BASE-A (step 0) → Arm A steps 106–848 → BASE-B (step 0) → Arm B steps 106–848.

**20 probes per checkpoint:** 14 core opinion/voice/fact probes (same as the probe notebook for
direct comparability) + 6 v4-val probes (3 grounded + 3 abstain held-out rows).
- Grounded probes include the actual source spans in the user turn — the model should ground its
  answer on those spans. The held-out reference answer is printed below each response.
- Abstain probes ask questions outside the model's scope. A good checkpoint declines in-voice rather
  than hallucinating.

**think\_leak fix:** `("<think>" in raw) or ("</think>" in raw)` — catches both the opening and
closing Qwen3 thinking tags.

**Crash-safe:** partial results land on Drive after every checkpoint; re-running skips already-scored
checkpoints. A preempted VM costs at most one checkpoint of generation work.

**~3–8 min per checkpoint on H100.** With 18 checkpoints total, budget ~2–3 h. Split into two Colab
sessions if needed: run BASE-A + Arm A in one session, BASE-B + Arm B in another.""")
code("PROBES = " + json.dumps(PROBES, ensure_ascii=False, indent=1))
code(r"""import re, json, glob, gc, torch, os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Copied verbatim from build_probe_nb.py for comparability
PROF    = re.compile(r"\b(sik|am[ıi]na|amc[ıi]k|g[öo]t|pi[çc]|o[çc]\b|orospu|yarr?a[kğ]|pezevenk|kahpe|ibne|gavat|anan[ıi]|avrad[ıi]|sokay[ıi]m|siktir|yavşak|şerefsiz)", re.I)
FOREIGN = re.compile(r"[一-鿿぀-ヿ가-힯Ѐ-ӿ؀-ۿ]")   # CJK/Kana/Hangul/Cyrillic/Arabic = hard collapse


def gen(model, tok, q):
    # Seeded sampling — identical params to probe notebook and v3sweep for direct comparability.
    torch.manual_seed(SEED)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    inp  = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out  = model.generate(
        input_ids=inp, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.85,
        repetition_penalty=1.2, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    g       = out[0][inp.shape[1]:]
    stopped = int(g[-1]) == tok.eos_token_id
    raw     = tok.decode(g, skip_special_tokens=False)
    return tok.decode(g, skip_special_tokens=True).strip(), stopped, raw


def run_probes_on(adir, label, base_model=None):
    # Load a checkpoint (or bare base when adir is None) and run all 20 probes.
    if adir is None:
        assert base_model is not None, "base_model required when adir is None"
        print(f"\n[{label}] loading BASE model ({base_model}) ...")
        m, tk = FastLanguageModel.from_pretrained(
            model_name=base_model, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    else:
        print(f"\n[{label}] loading {adir} ...")
        m, tk = FastLanguageModel.from_pretrained(
            model_name=adir, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    tk = get_chat_template(tk, chat_template="qwen-2.5", map_eos_token=True)
    FastLanguageModel.for_inference(m)

    rows = []
    for i, p in enumerate(PROBES):
        a, st, raw = gen(m, tk, p["q"])
        prof_c    = len(PROF.findall(a))
        foreign_c = len(FOREIGN.findall(a))
        # Fixed think-leak detector: catches both <think> and </think>
        think_lk  = ("<think>" in raw) or ("</think>" in raw)
        row = {
            "label": label, "kind": p["kind"], "topic": p["topic"], "q": p["q"],
            "answer": a, "prof": prof_c, "foreign": foreign_c,
            "think_leak": think_lk, "stopped": st,
        }
        if "ref" in p:
            row["ref"] = p["ref"]
        rows.append(row)
        print(f"  [{i+1:02d}/{len(PROBES)}] {p['kind']:<9s} {p['topic'][:28]:28s} "
              f"prof={prof_c} foreign={foreign_c} think={think_lk} stop={st}")
    del m, tk; gc.collect(); torch.cuda.empty_cache()
    return rows


def _step_num(path):
    import re as _re
    m_ = _re.search(r"_step(\d+)$", os.path.basename(path))
    return int(m_.group(1)) if m_ else -1


# ── Discover checkpoints ──────────────────────────────────────────────────────────────────────────
a_ckpts = sorted(glob.glob(f"{OUTPUT_ROOT}/qwen3_{RUN}_A_step*"), key=_step_num)
b_ckpts = sorted(glob.glob(f"{OUTPUT_ROOT}/qwen3_{RUN}_B_step*"), key=_step_num)
print(f"Arm A checkpoints ({len(a_ckpts)}): {[os.path.basename(c) for c in a_ckpts]}")
print(f"Arm B checkpoints ({len(b_ckpts)}): {[os.path.basename(c) for c in b_ckpts]}")
assert a_ckpts or b_ckpts, (
    f"No v5 checkpoints found under {OUTPUT_ROOT}/ — run Sections 6 and/or 7 first.")

# ── Crash-safe accumulator ────────────────────────────────────────────────────────────────────────
_PARTIAL  = f"{OUTPUT_ROOT}/speaker_v5_outputs.partial.json"
all_results = json.load(open(_PARTIAL, encoding="utf-8")) if os.path.exists(_PARTIAL) else []
_done = {r["label"] for r in all_results}
if _done:
    print(f"resuming: {len(_done)} checkpoints already scored in {_PARTIAL}: {sorted(_done)}")


def _score(adir, label, arm, step, base_model=None):
    if label in _done:
        print(f"[{label}] already scored — skipping")
        return
    all_results.append({"label": label, "arm": arm, "step": step,
                        "rows": run_probes_on(adir, label, base_model=base_model)})
    json.dump(all_results, open(_PARTIAL, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  [partial saved] {len(all_results)} checkpoints -> {_PARTIAL}")


# ── Run in order: BASE-A → Arm A → BASE-B → Arm B ───────────────────────────────────────────────
_score(None, "A_step0000_BASE", "A", 0, base_model=BASE_A)
for _c in a_ckpts:
    _score(_c, f"A_step{_step_num(_c):04d}", "A", _step_num(_c))
_score(None, "B_step0000_BASE", "B", 0, base_model=BASE_B)
for _c in b_ckpts:
    _score(_c, f"B_step{_step_num(_c):04d}", "B", _step_num(_c))

all_results.sort(key=lambda r: ({"A": 0, "B": 1}[r["arm"]], r["step"]))
print(f"\nGeneration complete: {len(all_results)} checkpoints x {len(PROBES)} probes")""")

# ============================================================ 9. WRITE OUTPUTS
md(r"""## 9. Write output files → `speaker_v5_outputs.{md,json}` to Drive

Re-runnable: rebuilds both files from `all_results` without re-running inference.

For **grounded** and **abstain** probes, the markdown also prints the held-out reference answer
so you can compare model output directly against the gold response.""")
code(r"""import json, os

_OUT_JSON = f"{OUTPUT_ROOT}/speaker_v5_outputs.json"
_OUT_MD   = f"{OUTPUT_ROOT}/speaker_v5_outputs.md"

# ── JSON (machine-readable) ───────────────────────────────────────────────────────────────────────
json.dump(all_results, open(_OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ── Markdown (human-readable — this is the primary deliverable) ──────────────────────────────────
_WARNING = (
    "**Auto stats (prof count, foreign-char count, think-leak, stopped) are degeneracy "
    "sanity checks ONLY — coherence/collapse is judged by READING. "
    "eval_loss does NOT detect Turkish morphology collapse. "
    "Do not rank checkpoints by any number in this file.**"
)
_ARM_DESC = {
    "A": "Arm A — abliterated (`huihui-ai/Huihui-Qwen3-14B-abliterated-v2`)",
    "B": "Arm B — stock (`Qwen/Qwen3-14B`)",
}

L = [
    f"# Speaker v5 outputs — {len(all_results)} checkpoints × {len(PROBES)} probes\n",
    _WARNING, "",
    ("Order: BASE-A (step 0) → Arm A steps 106–848 → BASE-B (step 0) → Arm B steps 106–848.  "
     "20 probes per checkpoint: 14 core + 3 grounded + 3 abstain held-out from v4 val."),
    ("For grounded/abstain probes the held-out reference answer is printed below each model response, "
     "labelled _reference (held-out val):_\n"),
    "---\n",
]

for _ckpt in all_results:
    _lbl  = _ckpt["label"]
    _arm  = _ckpt["arm"]
    _step = _ckpt["step"]
    _desc = _ARM_DESC.get(_arm, _arm)
    if _step == 0:
        L.append(f"## {_lbl} — BASE model / no adapter ({_desc})\n")
    else:
        L.append(f"## {_lbl} — {_desc}\n")

    for _r in _ckpt["rows"]:
        L.append(f"**[{_r['kind']}] {_r['topic']}**")
        L.append(f"Q: {_r['q'][:120]}{'...' if len(_r['q']) > 120 else ''}\n")
        for _line in _r["answer"].splitlines():
            L.append(f"> {_line}")
        L.append("")
        L.append(f"_stats: prof={_r['prof']} | foreign={_r['foreign']} | think_leak={_r['think_leak']} | stopped={_r['stopped']}_")
        if _r.get("ref"):
            L.append("")
            L.append("_reference (held-out val):_")
            for _line in _r["ref"].splitlines():
                L.append(f"> {_line}")
        L.append("")
    L.append("---\n")

open(_OUT_MD, "w", encoding="utf-8").write("\n".join(L))
print(f"WROTE  {_OUT_JSON}")
print(f"WROTE  {_OUT_MD}")
print("Download speaker_v5_outputs.md for a joint read to pick the winner checkpoint(s).")""")

# ============================================================ 10. EXPORT (GGUF)
md(r"""## 10. Export winner(s) → Q4_K_M GGUF + Ollama Modelfile

Set `WINNERS` to `[(arm, step), ...]` picks from the score file read, then run this cell.
Builds llama.cpp once; exports Q4_K_M only (what runs locally in Ollama).

**GGUF naming:** `speaker-qwen3-14b-v5-{arm}-step{N}-q4_k_m.gguf`
**Modelfile naming:** `Modelfile-speaker-v5-{arm}-step{N}`""")
code(r"""WINNERS = []   # e.g. [("A", 424), ("B", 530)] — fill from the outputs.md read, then run

import glob, shutil, subprocess, gc, os

def _sh(c):
    print("$", c)
    subprocess.run(c, shell=True, check=True)

LCPP = "/content/llama.cpp"
if WINNERS and not os.path.isdir(LCPP):
    _sh(f"git clone -q https://github.com/ggml-org/llama.cpp {LCPP}")
    _sh("pip install -q gguf sentencepiece protobuf")
    _sh(f"cmake -S {LCPP} -B {LCPP}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /content/_lcpp.log 2>&1")
    _sh(f"cmake --build {LCPP}/build -j --target llama-quantize >> /content/_lcpp.log 2>&1")

def _qbin():
    for _p in [f"{LCPP}/build/bin/llama-quantize", f"{LCPP}/build/llama-quantize"]:
        if os.path.exists(_p): return _p
    return glob.glob(f"{LCPP}/**/llama-quantize", recursive=True)[0]

TMPL = ("{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
        "{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n")


def export_winner(arm, step):
    from unsloth import FastLanguageModel
    _adir   = f"{OUTPUT_ROOT}/qwen3_{RUN}_{arm}_step{step:04d}"
    _base   = BASE_A if arm == "A" else BASE_B
    assert os.path.isdir(_adir), f"checkpoint not found: {_adir}"
    print("=" * 72, f"\nEXPORT Arm {arm} step {step:04d}")

    # Merge adapter into 16-bit base
    m2, t2  = FastLanguageModel.from_pretrained(model_name=_adir, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    _merged = f"/content/merged_v5_{arm}_step{step:04d}"
    m2.save_pretrained_merged(_merged, t2, save_method="merged_16bit")
    del m2, t2; gc.collect(); torch.cuda.empty_cache()

    # Convert to f16 GGUF then quantize to Q4_K_M
    _f16  = f"/content/v5_{arm}_step{step:04d}_f16.gguf"
    _gguf = f"speaker-qwen3-14b-v5-{arm}-step{step:04d}-q4_k_m.gguf"
    _sh(f'python {LCPP}/convert_hf_to_gguf.py "{_merged}" --outfile "{_f16}" --outtype f16')
    shutil.rmtree(_merged, ignore_errors=True)
    _sh(f'"{_qbin()}" "{_f16}" "{OUTPUT_ROOT}/{_gguf}" Q4_K_M')
    os.remove(_f16)

    # Write Ollama Modelfile
    _mf_name = f"Modelfile-speaker-v5-{arm}-step{step:04d}"
    _mf_lines = [
        f"FROM ./{_gguf}",
        'TEMPLATE ' + '"'*3 + TMPL + '"'*3,
        'SYSTEM '   + '"'*3 + SYS  + '"'*3,
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.85",
        "PARAMETER repeat_penalty 1.2",
        "PARAMETER num_ctx 4096",
        "PARAMETER num_predict 512",
        'PARAMETER stop "<|im_end|>"',
        'PARAMETER stop "<|endoftext|>"',
        'PARAMETER stop "<think>"',
    ]
    open(f"{OUTPUT_ROOT}/{_mf_name}", "w", encoding="utf-8").write("\n".join(_mf_lines) + "\n")
    print(f"WROTE  {OUTPUT_ROOT}/{_gguf}")
    print(f"WROTE  {OUTPUT_ROOT}/{_mf_name}")


if not WINNERS:
    print("WINNERS is empty — fill it from speaker_v5_outputs.md (e.g. WINNERS=[('A',424)]) and re-run.")
for _arm, _step in WINNERS:
    try:
        export_winner(_arm, _step)
    except Exception as _e:
        import traceback; traceback.print_exc()
        print(f"!! export failed Arm {_arm} step {_step}: {_e}")""")

# ============================================================ 11. WHAT TO SEND BACK
md(r"""## 11. What to send back

Download **`speaker_v5_outputs.md`** from Drive and read it carefully:

1. **BASE-A / BASE-B (step 0):** confirm both pretrained models speak coherent Turkish. If broken here,
   there is a loading issue, not a training issue.

2. **Arm A vs Arm B ascending steps:** look for the first step where answers stay coherent AND the
   voice (agzi bozuk, senli-benli, doğal küfürlü) is clearly present. The probe notebook established
   that LR 1e-4 collapses at 70–105 steps; at LR 1.5e-5 collapse should either not appear or appear
   much later — verify this.

3. **Grounded probes (kind=grounded):** the model should ground its answer in the provided spans, not
   hallucinate. Compare the model answer against the _reference (held-out val):_ line.

4. **Abstain probes (kind=abstain):** the model should decline in-voice, NOT answer. Compare against
   the reference decline.

5. **Arm A vs Arm B:** note any systematic difference in voice fidelity or collapse resilience.
   Pick the checkpoint (arm, step) to export; fill `WINNERS` in Section 10 and run.

Send the `.md` back for a joint read. Numbers (prof, foreign, think_leak, stopped) are degeneracy
signals only — coherence, morphological correctness, and voice fidelity are judged by reading.
""")

# ============================================================ WRITE NOTEBOOK
nb = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "H100"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = Path(__file__).with_name("qwen_v5_train.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"\nwrote {out}  ({len(CELLS)} cells, {len(PROBES)} probes)")
