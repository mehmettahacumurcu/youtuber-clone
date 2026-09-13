"""Generator for training/collapse_probe.ipynb — the collapse-onset probe.

Trains 2 arms to locate the step at which Turkish morphology collapse occurs and test
whether Turkish replay rescues it.

  Arm P: onset locator — r16/α32, LR 1e-4, max_steps=350, save every 35 steps (10 ckpts).
  Arm R: replay rescue — same recipe + CohereForAI/aya_dataset Turkish replay at
         REPLAY_FRAC=0.4 of rows, max_steps=606, save every 101 steps (6 ckpts).

No winner-picking, no GGUF export, no composite health leaderboard.
Human reads every probe answer at every saved checkpoint.

Run:  .venv\\Scripts\\python.exe -m training.build_probe_nb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.prompt import SYS as CANON_SYS  # noqa: E402

# ---- byte-identity guard: the notebook hardcodes SYS; it MUST equal rag.prompt.SYS (= dataset SYS)
SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")
assert SYS == CANON_SYS, "SYS drifted from rag.prompt.SYS — fix before generating"

# ---- 14 probes copied verbatim from cell 17 of qwen_voice_sft.ipynb (v3sweep) for comparability
PROBES = [
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

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})


# ============================================================ TITLE
md(r"""# Speaker collapse probe — onset locator (Arm P) + replay rescue (Arm R)

**Purpose:** locate the step at which Turkish morphology collapse begins (Arm P) and test whether
generic-Turkish replay prevents it (Arm R). **No winner-picking, no GGUF export, no auto-quality
leaderboard.** This notebook produces dense adapter checkpoints and a human-readable `.md` of every
probe answer at every checkpoint. Collapse vs coherence is judged by **READING** the outputs.

| arm | purpose | r/α | LR | max\_steps | save every | checkpoints |
|---|---|---|---|---|---|---|
| **P** | onset locator | 16/32 | 1e-4 | 350 | 35 steps | 10 (steps 35–350) |
| **R** | replay rescue | 16/32 | 1e-4 | 606 | 101 steps | 6 (steps 101–606) |

- **Base model:** `huihui-ai/Huihui-Qwen3-14B-abliterated-v2` (A/B alternative `Qwen/Qwen3-14B` — one-line swap in Section 4).
- **Data:** `voice_sft_v3_train.jsonl` (2,428 rows) — same as the v3sweep, enabling direct comparison.
- **Hardware:** H100 recommended (~1–2 h per arm). A100/L4 also fine.
- Robust: each arm saves adapters to Drive first; the generation pass (Section 8) rediscovers them and is separately re-runnable.
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
RUN = "probe1"   # tags every output; never clobbers v3sweep / v4 files on Drive
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print("outputs ->", OUTPUT_ROOT, "| run tag:", RUN)""")

# ============================================================ 3. DATA UPLOAD
md(r"""## 3. Data — upload `voice_sft_v3_train.jsonl` + `voice_sft_val.jsonl`

Same dataset as the v3sweep run — the probe compares directly against those sweep results.
Right-click → Upload in the Colab file panel, or uncomment the git-clone block below.""")
code(r"""import glob as _glob, json as _json

def _find(name):
    for d in [".", "/content", "/content/drive/MyDrive"]:
        p = os.path.join(d, name)
        if os.path.isfile(p): return p
    hits = _glob.glob(f"/content/**/{name}", recursive=True) or _glob.glob(f"**/{name}", recursive=True)
    return hits[0] if hits else None

# Uncomment to clone repo and pull files instead of manual upload:
# import getpass; PAT = getpass.getpass("GitHub PAT: ")
# !git clone --depth 1 https://{PAT}@github.com/mehmettahacumurcu/youtuber-clone.git _repo
# TRAIN = "_repo/data/dataset/voice_sft_v3_train.jsonl"; VAL = "_repo/data/dataset/voice_sft_val.jsonl"

TRAIN = _find("voice_sft_v3_train.jsonl"); VAL = _find("voice_sft_val.jsonl")
assert TRAIN and VAL, "voice_sft_v3_train.jsonl / voice_sft_val.jsonl not found — upload them."
_rows = [_json.loads(l) for l in open(TRAIN, encoding="utf-8") if l.strip()]
assert len(_rows) >= 2000, f"only {len(_rows)} rows — expected 2,428 (v3 full set); wrong file?"
print(f"train: {TRAIN} ({len(_rows)} rows) | val: {VAL}")""")

# ============================================================ 4. CONFIG
md("## 4. Config — constants and arm parameters")
code(r"""# Base model. To A/B against stock Qwen3-14B, change to "Qwen/Qwen3-14B" — one-line swap.
BASE         = "huihui-ai/Huihui-Qwen3-14B-abliterated-v2"
MAX_SEQ_LEN  = 4096
TARGETS      = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # all 7 linear
LORA_DROPOUT = 0.05
SEED         = 7
BS, GA       = 2, 4        # eff-batch 8; drop to 1,8 if you OOM

# Arm P — onset locator: dense saves to pinpoint WHERE collapse begins
P_MAX_STEPS  = 350
P_SAVE_EVERY = 35          # 10 checkpoints: steps 35, 70, 105 ... 350

# Arm R — replay rescue: longer run with Turkish replay mixed in
R_MAX_STEPS  = 606
R_SAVE_EVERY = 101         # 6 checkpoints: steps 101, 202, 303, 404, 505, 606
REPLAY_FRAC  = 0.4         # 40% of the combined training dataset rows will be Turkish replay rows

# Persona system prompt — byte-identical to the dataset system prompt and rag/prompt.py SYS.
SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")

# Neutral replay system prompt — intentionally NOT the persona SYS.
# Replay teaches Turkish morphology/grammar, NOT the character. This separation is critical.
REPLAY_SYS = "Sana bir soru sorulur; net ve dogru Turkce yanit ver."

print("BASE:", BASE)
print(f"Arm P: max_steps={P_MAX_STEPS}, save_every={P_SAVE_EVERY} -> {P_MAX_STEPS // P_SAVE_EVERY} checkpoints")
print(f"Arm R: max_steps={R_MAX_STEPS}, save_every={R_SAVE_EVERY}, replay_frac={REPLAY_FRAC} -> {R_MAX_STEPS // R_SAVE_EVERY} checkpoints")""")

# ============================================================ 5. HELPERS
md(r"""## 5. Helpers — load_fresh, make_peft, fmt_ds, StepSaver callback""")
code(r"""import torch, time, gc
from datasets import load_dataset, Dataset
from transformers import TrainerCallback

def load_fresh():
    # Load a clean 4-bit base model + Qwen-2.5 thinking-free chat template each time.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    tok = get_chat_template(tok, chat_template="qwen-2.5", map_eos_token=True)  # plain thinking-free ChatML
    tok.truncation_side = "left"   # overflow trims the FRONT (context/spans), never the answer+<|im_end|>
    return model, tok

def make_peft(model):
    # Attach r=16/alpha=32 LoRA and assert embeddings/lm_head stay frozen (both arms use same rank).
    from unsloth import FastLanguageModel
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=TARGETS, use_gradient_checkpointing="unsloth", random_state=SEED)
    for n_, p_ in model.named_parameters():          # anti-forgetting: embeddings/head must stay frozen
        if ("embed_tokens" in n_ or "lm_head" in n_) and p_.requires_grad:
            raise RuntimeError(f"embeddings trainable ({n_}) -- must be frozen")
    return model

def fmt_ds(raw_ds, tok):
    # Apply chat template to a dataset that has a 'messages' column.
    def _fmt(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}
    return raw_ds.map(_fmt)


class StepSaver(TrainerCallback):
    # Saves LoRA adapter + tokenizer to Drive every `save_every` optimizer steps.
    def __init__(self, arm, save_every, tok, saved_dict):
        self.arm = arm
        self.save_every = save_every
        self.tok = tok
        self.saved = saved_dict

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step > 0 and state.global_step % self.save_every == 0:
            p = f"{OUTPUT_ROOT}/qwen3_{RUN}_{self.arm}_step{state.global_step:04d}"
            model.save_pretrained(p)
            self.tok.save_pretrained(p)
            self.saved[state.global_step] = p
            print(f"  [ckpt] {self.arm} step{state.global_step:04d} -> {p}")""")

# ============================================================ 6. ARM P
md(r"""## 6. Arm P — onset locator (max_steps=350, save every 35 steps)

Runs the same recipe as the v3sweep `gentle` arm (r16/α32, LR 1e-4) but stops at 350 steps with
dense saves every 35. Collapse is expected somewhere between step 105 (run2 was healthy at ~70–105
steps) and step 303 (v3sweep was dead by epoch-1, ~303 steps). The 10 checkpoints bracket the onset.

**~40–60 min on H100.**""")
code(r"""# Invariant: print and verify step math before training
_train_n   = sum(1 for _l in open(TRAIN, encoding="utf-8") if _l.strip())
_eff_b     = BS * GA   # 8
_steps_per_epoch_P = max(1, _train_n // _eff_b)
_ep_equiv_P        = P_MAX_STEPS / _steps_per_epoch_P
print(f"ARM P: {_train_n} rows, eff-batch {_eff_b}, max_steps={P_MAX_STEPS} (~{_ep_equiv_P:.2f} epochs)")
print(f"       save_every={P_SAVE_EVERY} -> checkpoints at steps: {list(range(P_SAVE_EVERY, P_MAX_STEPS + 1, P_SAVE_EVERY))}")
assert P_MAX_STEPS % P_SAVE_EVERY == 0, "P_MAX_STEPS must be divisible by P_SAVE_EVERY for complete coverage"
assert _ep_equiv_P < 3.0, f"ARM P epoch equivalent {_ep_equiv_P:.2f} unexpectedly large — check row count/eff-batch"
assert _ep_equiv_P > 0.5, f"ARM P epoch equivalent {_ep_equiv_P:.2f} suspiciously small — check row count"
print("ARM P step-math asserts passed.")""")
code(r"""from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

model_P, tok_P = load_fresh()
model_P        = make_peft(model_P)
train_ds_P     = fmt_ds(load_dataset("json", data_files=TRAIN, split="train"), tok_P)
val_ds_P       = fmt_ds(load_dataset("json", data_files=VAL,   split="train"), tok_P)

saved_P = {}
saver_P = StepSaver("P", P_SAVE_EVERY, tok_P, saved_P)

args_P = SFTConfig(
    output_dir="/content/out_P", dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
    per_device_train_batch_size=BS, gradient_accumulation_steps=GA,
    max_steps=P_MAX_STEPS,          # max_steps overrides num_train_epochs when > 0
    learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
    weight_decay=0.01, optim="adamw_8bit", logging_steps=10,
    eval_strategy="no", save_strategy="no", seed=SEED, report_to="none",
    fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported())
try:
    tr_P = SFTTrainer(model=model_P, tokenizer=tok_P, train_dataset=train_ds_P,
                      eval_dataset=val_ds_P, args=args_P, callbacks=[saver_P])
except TypeError:
    tr_P = SFTTrainer(model=model_P, processing_class=tok_P, train_dataset=train_ds_P,
                      eval_dataset=val_ds_P, args=args_P, callbacks=[saver_P])
tr_P = train_on_responses_only(tr_P, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

t0 = time.time()
tr_P.train()
# Safety pass: on_step_end fires AT end of step; if the last step is P_MAX_STEPS it should save,
# but defend against edge cases in case the trainer ends before the callback fires.
if P_MAX_STEPS not in saved_P:
    _p = f"{OUTPUT_ROOT}/qwen3_{RUN}_P_step{P_MAX_STEPS:04d}"
    model_P.save_pretrained(_p); tok_P.save_pretrained(_p); saved_P[P_MAX_STEPS] = _p
    print(f"  [safety-save] P step{P_MAX_STEPS:04d} -> {_p}")
print(f"\nARM P DONE: {(time.time()-t0)/60:.1f} min | {len(saved_P)} checkpoints: {sorted(saved_P)}")
del tr_P, model_P, tok_P; gc.collect(); torch.cuda.empty_cache()""")

# ============================================================ 7. ARM R
md(r"""## 7. Arm R — replay rescue (max_steps=606, save every 101 steps)

Mixes the v3 train set with Turkish rows from `CohereForAI/aya_dataset` at `REPLAY_FRAC=0.4`
(40 % of combined dataset rows). Replay rows carry `REPLAY_SYS` — NEVER the persona `SYS`.
This separation is intentional and critical: replay teaches Turkish morphology/grammar, not the
character. Arm P's collapse onset calibrates what "rescued" means here.

**~60–90 min on H100.**""")
code(r"""import math, random, json as _json_r
from datasets import load_dataset as _lds, concatenate_datasets

# ── Step math for Arm R ──────────────────────────────────────────────────────────────────────────
_eff_b_R      = BS * GA   # 8
_train_n_R    = sum(1 for _l in open(TRAIN, encoding="utf-8") if _l.strip())
_ep_equiv_R   = (R_MAX_STEPS * _eff_b_R) / _train_n_R   # approximate (before mixing)
# How many replay rows to include so they are exactly REPLAY_FRAC of the combined dataset:
#   replay / (voice + replay) = REPLAY_FRAC  =>  replay = voice * FRAC / (1 - FRAC)
_replay_n_need = math.ceil(_train_n_R * REPLAY_FRAC / (1.0 - REPLAY_FRAC))
# For reference: the trainer will consume R_MAX_STEPS * eff_batch = 606*8 = 4848 row-instances;
# of those, ~REPLAY_FRAC*4848 = ~1939 will be replay instances (via dataset cycling).
print(f"ARM R: {_train_n_R} voice rows, eff-batch {_eff_b_R}, max_steps={R_MAX_STEPS} (~{_ep_equiv_R:.2f} voice-epoch equiv)")
print(f"       replay rows needed for REPLAY_FRAC={REPLAY_FRAC}: {_replay_n_need}")
print(f"       trainer will consume ~{R_MAX_STEPS * _eff_b_R} row-instances (~{int(R_MAX_STEPS * _eff_b_R * REPLAY_FRAC)} replay instances)")
assert R_MAX_STEPS % R_SAVE_EVERY == 0, "R_MAX_STEPS must be divisible by R_SAVE_EVERY for complete coverage"

# ── Load Turkish subset of aya_dataset ───────────────────────────────────────────────────────────
print("\nLoading CohereForAI/aya_dataset ...")
_aya = _lds("CohereForAI/aya_dataset", split="train")
print(f"  columns: {_aya.column_names}")
# Robustly find the language column — inspect actual schema at runtime and filter accordingly
_lang_col = None
for _c in _aya.column_names:
    if "lang" in _c.lower():
        _lang_col = _c; break
assert _lang_col is not None, f"No language column in aya_dataset — columns: {_aya.column_names}"
print(f"  language column: '{_lang_col}'")
_sample_langs = list(set(_aya.select(range(min(500, len(_aya))))[_lang_col]))
print(f"  sample language values: {sorted(_sample_langs)[:30]}")
# Common Turkish identifiers; extend _tur_ids if the print above shows a different form
_tur_ids = {"Turkish", "tr", "tur", "Türkçe", "turkish"}
_aya_tr = _aya.filter(lambda ex: ex[_lang_col] in _tur_ids)
print(f"  Turkish rows found: {len(_aya_tr)}")
assert len(_aya_tr) > 0, (
    f"No Turkish rows found (lang_col='{_lang_col}', tried ids={_tur_ids}). "
    "Check the sample above and add the correct identifier to _tur_ids.")

# ── Sample replay rows (with replacement if needed) ──────────────────────────────────────────────
random.seed(SEED)
if len(_aya_tr) >= _replay_n_need:
    _replay_indices = random.sample(range(len(_aya_tr)), _replay_n_need)
    print(f"  sampled {_replay_n_need} replay rows WITHOUT replacement")
else:
    _rep_factor = _replay_n_need / len(_aya_tr)
    print(f"  WARNING: Turkish subset ({len(_aya_tr)} rows) < needed ({_replay_n_need}). "
          f"Sampling WITH REPLACEMENT (repetition factor: {_rep_factor:.2f}x). "
          "Consider filtering by a stricter quality column if available.")
    _replay_indices = [random.choice(range(len(_aya_tr))) for _ in range(_replay_n_need)]

# ── Convert to chat format with NEUTRAL system prompt ────────────────────────────────────────────
# Replay uses REPLAY_SYS, never the persona SYS. This separation is intentional and critical:
# replay rows must NOT carry the persona system prompt — they teach Turkish, not the character.
# Inspect actual column names from the aya_dataset schema (printed above).
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
# 1. Replay rows must NOT carry the persona SYS
for _rr in _replay_rows[:20]:
    assert _rr["messages"][0]["content"] != SYS, \
        "BUG: replay row carries the persona SYS — must use REPLAY_SYS"
print("  assert: replay rows carry REPLAY_SYS (not persona SYS) [OK]")

# 2. Realized replay fraction must be within ±2% of REPLAY_FRAC
_combined_n   = _train_n_R + len(_replay_rows)
_real_frac    = len(_replay_rows) / _combined_n
assert abs(_real_frac - REPLAY_FRAC) <= 0.02, (
    f"Realized replay fraction {_real_frac:.3f} is outside ±2% of REPLAY_FRAC={REPLAY_FRAC}. "
    f"Voice rows={_train_n_R}, replay rows={len(_replay_rows)}, combined={_combined_n}.")
print(f"  assert: realized replay fraction {_real_frac:.3f} within ±2% of {REPLAY_FRAC} [OK]")

print(f"\nReplay construction done: {_train_n_R} voice + {len(_replay_rows)} replay = {_combined_n} combined rows")""")
code(r"""from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# ── Load voice rows and build the shuffled mixed dataset ─────────────────────────────────────────
_voice_rows = [_json_r.loads(_l) for _l in open(TRAIN, encoding="utf-8") if _l.strip()]
_mixed_raw  = _voice_rows + _replay_rows
random.seed(SEED)
random.shuffle(_mixed_raw)
_mixed_ds = Dataset.from_list(_mixed_raw)
print(f"Mixed dataset: {len(_mixed_ds)} rows ({len(_voice_rows)} voice + {len(_replay_rows)} replay), shuffled (seed={SEED})")

model_R, tok_R = load_fresh()
model_R        = make_peft(model_R)
train_ds_R     = fmt_ds(_mixed_ds, tok_R)
val_ds_R       = fmt_ds(load_dataset("json", data_files=VAL, split="train"), tok_R)
print(f"Steps per mixed-dataset pass: {len(train_ds_R) // (BS * GA)}")

saved_R = {}
saver_R = StepSaver("R", R_SAVE_EVERY, tok_R, saved_R)

args_R = SFTConfig(
    output_dir="/content/out_R", dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
    per_device_train_batch_size=BS, gradient_accumulation_steps=GA,
    max_steps=R_MAX_STEPS,          # max_steps overrides num_train_epochs when > 0
    learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
    weight_decay=0.01, optim="adamw_8bit", logging_steps=10,
    eval_strategy="no", save_strategy="no", seed=SEED, report_to="none",
    fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported())
try:
    tr_R = SFTTrainer(model=model_R, tokenizer=tok_R, train_dataset=train_ds_R,
                      eval_dataset=val_ds_R, args=args_R, callbacks=[saver_R])
except TypeError:
    tr_R = SFTTrainer(model=model_R, processing_class=tok_R, train_dataset=train_ds_R,
                      eval_dataset=val_ds_R, args=args_R, callbacks=[saver_R])
tr_R = train_on_responses_only(tr_R, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

t0 = time.time()
tr_R.train()
if R_MAX_STEPS not in saved_R:
    _p = f"{OUTPUT_ROOT}/qwen3_{RUN}_R_step{R_MAX_STEPS:04d}"
    model_R.save_pretrained(_p); tok_R.save_pretrained(_p); saved_R[R_MAX_STEPS] = _p
    print(f"  [safety-save] R step{R_MAX_STEPS:04d} -> {_p}")
print(f"\nARM R DONE: {(time.time()-t0)/60:.1f} min | {len(saved_R)} checkpoints: {sorted(saved_R)}")
del tr_R, model_R, tok_R; gc.collect(); torch.cuda.empty_cache()""")

# ============================================================ 8. GENERATE PROBE OUTPUTS
md(r"""## 8. Generate probe outputs from every checkpoint

Re-runnable from a fresh runtime — rediscovers checkpoints from Drive. Also probes the BASE model
with no adapter as a step-0 reference. **Crash-safe:** partial results are written to Drive after
every checkpoint (`speaker_probe1_outputs.partial.json`); re-running this cell skips checkpoints
already scored, so a preempted VM costs at most one checkpoint of generation work.

**Checkpoint order:** base (step 0) → Arm P steps 35–350 → Arm R steps 101–606.

Decoding parameters are seeded and identical to the v3sweep scoring cell for direct comparability.
**~3–8 min per checkpoint on H100.** (~2h total for all 17 checkpoints.)""")
code("PROBES = " + json.dumps(PROBES, ensure_ascii=False, indent=1))
code(r"""import re, json, glob, gc, torch, os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Copied verbatim from v3sweep cell 17
PROF    = re.compile(r"\b(sik|am[ıi]na|amc[ıi]k|g[öo]t|pi[çc]|o[çc]\b|orospu|yarr?a[kğ]|pezevenk|kahpe|ibne|gavat|anan[ıi]|avrad[ıi]|sokay[ıi]m|siktir|yavşak|şerefsiz)", re.I)
FOREIGN = re.compile(r"[一-鿿぀-ヿ가-힯Ѐ-ӿ؀-ۿ]")   # CJK/Kana/Hangul/Cyrillic/Arabic = hard collapse


def gen(model, tok, q):
    # Seeded sampling -- identical to v3sweep cell 17 for direct comparability.
    torch.manual_seed(SEED)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    inp = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(
        input_ids=inp, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.85,
        repetition_penalty=1.2, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    g = out[0][inp.shape[1]:]
    stopped = int(g[-1]) == tok.eos_token_id
    return tok.decode(g, skip_special_tokens=True).strip(), stopped, tok.decode(g, skip_special_tokens=False)


def run_probes_on(adir, label):
    # Load a checkpoint (or bare BASE when adir is None), run all 14 probes, return rows.
    if adir is None:
        print(f"\n[{label}] loading BASE model (no adapter) ...")
        m, tk = FastLanguageModel.from_pretrained(
            model_name=BASE, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
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
        think_lk  = "<think>" in raw
        rows.append({
            "label": label, "kind": p["kind"], "topic": p["topic"], "q": p["q"],
            "answer": a, "prof": prof_c, "foreign": foreign_c,
            "think_leak": think_lk, "stopped": st,
        })
        print(f"  [{i+1:02d}/{len(PROBES)}] {p['topic'][:28]:28s} "
              f"prof={prof_c} foreign={foreign_c} think={think_lk} stop={st}")
    del m, tk; gc.collect(); torch.cuda.empty_cache()
    return rows


def _step_num(path):
    # Extract step number from a checkpoint dir name like qwen3_probe1_P_step0035.
    import re as _re
    m_ = _re.search(r"_step(\d+)$", os.path.basename(path))
    return int(m_.group(1)) if m_ else -1


# ── Discover checkpoints ──────────────────────────────────────────────────────────────────────────
p_ckpts = sorted(glob.glob(f"{OUTPUT_ROOT}/qwen3_{RUN}_P_step*"), key=_step_num)
r_ckpts = sorted(glob.glob(f"{OUTPUT_ROOT}/qwen3_{RUN}_R_step*"), key=_step_num)
print(f"Arm P checkpoints ({len(p_ckpts)}): {[os.path.basename(c) for c in p_ckpts]}")
print(f"Arm R checkpoints ({len(r_ckpts)}): {[os.path.basename(c) for c in r_ckpts]}")
assert p_ckpts or r_ckpts, (
    f"No probe1 checkpoints found under {OUTPUT_ROOT}/ — run Sections 6 and 7 first.")

# ── Crash-safe accumulator: partial results land on Drive after EVERY checkpoint, and a rerun
# skips checkpoints already scored — a preempted VM costs at most one checkpoint of work.
_PARTIAL = f"{OUTPUT_ROOT}/speaker_probe1_outputs.partial.json"
all_results = json.load(open(_PARTIAL, encoding="utf-8")) if os.path.exists(_PARTIAL) else []
_done = {r["label"] for r in all_results}
if _done:
    print(f"resuming: {len(_done)} checkpoints already scored in {_PARTIAL}: {sorted(_done)}")


def _score(adir, label, arm, step):
    if label in _done:
        print(f"[{label}] already scored — skipping")
        return
    all_results.append({"label": label, "arm": arm, "step": step,
                        "rows": run_probes_on(adir, label)})
    json.dump(all_results, open(_PARTIAL, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  [partial saved] {len(all_results)} checkpoints -> {_PARTIAL}")


# ── Run in order: base (step 0) → Arm P ascending → Arm R ascending ─────────────────────────────
_score(None, "step0_BASE", "base", 0)                                  # coherence reference
for _c in p_ckpts:
    _score(_c, f"P_step{_step_num(_c):04d}", "P", _step_num(_c))
for _c in r_ckpts:
    _score(_c, f"R_step{_step_num(_c):04d}", "R", _step_num(_c))

all_results.sort(key=lambda r: ({"base": 0, "P": 1, "R": 2}[r["arm"]], r["step"]))
print(f"\nGeneration complete: {len(all_results)} checkpoints x {len(PROBES)} probes")""")

# ============================================================ 9. WRITE OUTPUTS
md(r"""## 9. Write output files → `speaker_probe1_outputs.{md,json}` to Drive

Re-runnable: rebuilds both files from `all_results` without re-running inference.""")
code(r"""import json, os

_OUT_JSON = f"{OUTPUT_ROOT}/speaker_probe1_outputs.json"
_OUT_MD   = f"{OUTPUT_ROOT}/speaker_probe1_outputs.md"

# ── JSON (machine-readable, full data) ───────────────────────────────────────────────────────────
json.dump(all_results, open(_OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ── Markdown (human-readable — this is the deliverable to send back) ─────────────────────────────
_WARNING = (
    "**Auto stats (prof count, foreign-char count, think-leak, stopped) are degeneracy "
    "sanity checks ONLY — coherence/collapse is judged by READING. "
    "Do not rank checkpoints by any number in this file.**"
)
_ARM_DESC = {
    "base": "BASE model / no adapter (step 0 reference)",
    "P":    "Arm P — onset locator",
    "R":    "Arm R — replay rescue",
}

L = [
    f"# Speaker probe1 outputs — {len(all_results)} checkpoints × {len(PROBES)} probes\n",
    _WARNING, "",
    "Order: base (step 0) → Arm P steps 35–350 → Arm R steps 101–606.",
    "Each H2 section = one checkpoint. Per probe: question, answer blockquote, stats line.\n",
    "---\n",
]

for _ckpt in all_results:
    _lbl  = _ckpt["label"]
    _arm  = _ckpt["arm"]
    _step = _ckpt["step"]
    L.append(f"## {_lbl} — {_ARM_DESC[_arm]}\n")
    for _r in _ckpt["rows"]:
        L.append(f"**[{_r['kind']}] {_r['topic']}**")
        L.append(f"Q: {_r['q']}\n")
        # raw answer as a blockquote
        for _line in _r["answer"].splitlines():
            L.append(f"> {_line}")
        L.append("")
        L.append(f"_stats: prof={_r['prof']} | foreign={_r['foreign']} | think_leak={_r['think_leak']} | stopped={_r['stopped']}_\n")
    L.append("---\n")

open(_OUT_MD, "w", encoding="utf-8").write("\n".join(L))
print(f"WROTE  {_OUT_JSON}")
print(f"WROTE  {_OUT_MD}")
print("Download speaker_probe1_outputs.md and read it to locate the collapse onset and judge replay effectiveness.")""")

# ============================================================ 10. WHAT TO SEND BACK
md(r"""## 10. What to send back

Download **`speaker_probe1_outputs.md`** from Drive and read it carefully:

1. **Arm P (steps 35–350):** find the lowest step where answers become word-salad or morphologically broken.
   That step is the collapse onset. Steps before it reveal the usable step budget.
2. **Arm R (steps 101–606):** compare at the same step ranges as Arm P.
   If Arm R stays coherent past Arm P's collapse onset, Turkish replay is rescuing it — and by how much.
3. **Base (step 0):** confirms the pretrained model is coherent in Turkish. If it is already broken here,
   there is a base-model loading issue, not a training issue.

Send the `.md` back for a joint read. Numbers in the stats lines are degeneracy signals only — coherence,
morphological correctness, and voice fidelity are judged by reading every answer.
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
out = Path(__file__).with_name("collapse_probe.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"wrote {out}  ({len(CELLS)} cells, {len(PROBES)} probes)")
