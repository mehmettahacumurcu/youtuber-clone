"""Generate training/qlora_train_multi.ipynb — one Run-all trains 3 base models sequentially.

Kept separate from the source notebook so the cell text stays readable/diffable. Re-run after
editing the CELLS list:  python training/build_multi_nb.py
"""
from __future__ import annotations
import json
from pathlib import Path

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
def code(text): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}

CELLS = []

CELLS.append(md(r"""
# Speaker voice-clone — QLoRA, **3 base models in one run** (Unsloth)

**Run-all trains every BASE model not already on Drive (idempotent), each saved at epochs 2/3/4/5.**
Models run one after another in a single runtime so you can compare *which base* and *which dose*
speaks most like him:

1. **Cosmos-Turkish-8B** (`ytu-ce-cosmos/Turkish-Llama-8b-v0.1`) — Llama-3 continued-pretrained on Turkish. *(already trained → auto-skipped)*
2. **Llama-3.1-8B base** (`unsloth/Meta-Llama-3.1-8B-bnb-4bit`) — the token-efficient control; **trained first** (proven clean).
3. **Gemma-2-9B base** (`unsloth/gemma-2-9b-bnb-4bit`) — **trained last**, with OOM-hardened eval (it failed at eval once).

**Recipe (from `finetune_plan` / CLAUDE.md, scaled for the full ~3.9M-token corpus):**
- QLoRA: 4-bit NF4 base + 16-bit LoRA. r=64, alpha=128, dropout=0.05, **all 7 linear projections**.
- **FROZEN embed_tokens / lm_head** — the biggest anti-forgetting lever (asserted each model).
- Objective: raw-text completion. Per-video docs tokenized + **EOS-joined** + pre-packed into 2048-token
  blocks. NO chat template, NO loss masking.
- LR 2e-4, cosine + warmup 0.05, **5 epochs (adapter saved at epochs 2/3/4/5)**, **effective batch 32**
  (per-device batch auto-scales to the GPU — H100 = bs8×ga4), wd 0.01, adamw_8bit. Val loss each epoch.
  *"Best model fast": comparability with cosmos is intentionally dropped — we optimize for speed/quality.*

**EOS contract (this is your "model stops when it wants"):** the dataset file carries NO literal EOS
(a base model would learn to print the string). This notebook appends `tokenizer.eos_token` between
docs in the packed stream, so each model learns to EMIT a real stop token at doc boundaries. Section 7
checks whether each trained model actually stops on its own. (Further *turn-boundary* shaping is the
later instruct step — not needed for this base-style run.)

**Run this in the Colab *web* UI (browser), NOT the VS Code extension.** The VS Code↔Colab bridge
drops on long unattended jobs and can take the runtime down with it (that's how we lost a llama run).
The browser session is far more stable for multi-hour training.

**Runtime: pick the biggest GPU you can get** — G4 (Blackwell, 96 GB) ≥ H100 (80 GB) > A100 > L4 > T4;
the per-device batch auto-scales to it. **Section 9's 9B fp16 merge needs ~18 GB system RAM:** G4 already
has ~176 GB (no toggle needed); on H100/A100/T4 **tick "High-RAM"** (a standard ~13 GB runtime OOMs the
merge — the only thing that's failed before).

**What you do:** run the 3 interactive cells at the top once (HF login → Drive mount → data), then
**Runtime ▸ Run all** and walk away. Each epoch-3/4/5 adapter is written to Drive the moment that
epoch ends, so a disconnect never loses finished work — and **any model whose adapters are already on
Drive is skipped**, so re-running after a drop (or after cosmos already finished) only trains what's
missing. If one model errors, the others still run.

**Hardware / time:** **H100-80 ≈ 1–1.5 h** for all three (fastest; the per-device batch auto-scales up
to use its 80 GB); A100-40 ≈ 2–3 h; T4 ≈ 6–9 h (works, slower; Gemma-2-9B is the heavy one). Effective
batch is held at 16 on every GPU so the recipe is identical — only wall-clock changes. Each model's
epoch-3/4/5 adapters land on Drive as it goes, so a disconnect is safe.
"""))

CELLS.append(md("## 1. Install (Unsloth + deps)"))
CELLS.append(code(r"""
%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    !pip install --no-deps bitsandbytes accelerate xformers peft trl triton
    !pip install --no-deps unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth

# --- FROZEN FALLBACK (reproducible Nov-2024 stack; TRL 0.12 API). If you use this, in the trainer
#     cell change `max_length=`->`max_seq_length=` and `processing_class=`->`tokenizer=`:
# %pip install -q "unsloth==2024.11.9" "unsloth_zoo==2024.11.7" "trl==0.12.2" "peft==0.13.2" \
#   "transformers==4.46.3" "datasets>=3.0,<4" "accelerate>=1.0,<1.2" "bitsandbytes>=0.44"
"""))

CELLS.append(md(r"""
## 2. Auth + persistence  ·  **interactive — do this once**

- **HF login**: Gemma-2 and the Cosmos/Llama bases live behind a license click. Accept each model's
  license on huggingface.co with your account first (one-time), then paste a *read* token here.
- **Drive mount**: where the 3 finished adapters land so they survive the runtime. Click-through once.
"""))
CELLS.append(code(r"""
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
      "| bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)

# --- Hugging Face login (needed for the gated bases) ---
from huggingface_hub import login
login()  # paste a READ token; accept each model's license on its HF page first

# --- Persist outputs to Drive (recommended). Set USE_DRIVE=False to keep them on the ephemeral runtime. ---
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = "/content/drive/MyDrive/speaker_models"
else:
    OUTPUT_ROOT = "/content/speaker_models"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print("adapters will be saved under:", OUTPUT_ROOT)
"""))

CELLS.append(md(r"""
## 3. Data  ·  **interactive — upload or clone train/val**

Either right-click `data/dataset/train.jsonl` + `val.jsonl` in VS Code → *Upload to Colab*, **or**
uncomment the git-clone block (needs a GitHub PAT).
"""))
CELLS.append(code(r"""
import glob
_SEARCH = [".", "/content", os.path.expanduser("~"), "/content/drive/MyDrive"]

def _find(name):
    for d in _SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    hits = glob.glob(f"/content/**/{name}", recursive=True) or glob.glob(f"**/{name}", recursive=True)
    return hits[0] if hits else None

TRAIN, VAL = _find("train.jsonl"), _find("val.jsonl")

# --- git-clone alternative (committed to the repo) ---
# import getpass; PAT = getpass.getpass("GitHub PAT: ")
# !git clone --depth 1 https://{PAT}@github.com/mehmettahacumurcu/youtuber-clone.git _repo
# TRAIN, VAL = "_repo/data/dataset/train.jsonl", "_repo/data/dataset/val.jsonl"

assert TRAIN and VAL, "train.jsonl / val.jsonl not found — upload them or use the git-clone block above."
print("train:", TRAIN, os.path.getsize(TRAIN), "B | val:", VAL, os.path.getsize(VAL), "B")
"""))

CELLS.append(md(r"""
## 4. Config — the three models + shared hyperparameters

All three share one recipe so the comparison is apples-to-apples. EOS, tokenization and packing are
resolved **per model** at train time (each tokenizer differs).
"""))
CELLS.append(code(r"""
# BASE models only (never Instruct — RLHF sanitizes profanity, which we must keep). A model whose
# final-epoch adapter is ALREADY on Drive is skipped (idempotent), so re-running after a disconnect —
# or after cosmos already finished — only trains what's missing. Order matters now: llama BEFORE gemma,
# because llama trained cleanly already and gemma OOM'd once — do the proven one first so a gemma
# re-failure can never cost us llama again.
MODELS = [
    {"name": "cosmos-turkish-8b", "model_name": "ytu-ce-cosmos/Turkish-Llama-8b-v0.1"},  # done -> auto-skipped
    {"name": "llama-3.1-8b",      "model_name": "unsloth/Meta-Llama-3.1-8B-bnb-4bit"},    # proven; train first
    {"name": "gemma-2-9b",        "model_name": "unsloth/gemma-2-9b-bnb-4bit"},           # OOM'd once; train last
]

# --- shared QLoRA hyperparameters (full-corpus recipe) ---
MAX_SEQ_LEN  = 2048
LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05
TARGETS      = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EPOCHS       = 5            # run the full 5; we keep per-epoch checkpoints to compare doses
SAVE_EPOCHS  = (2, 3, 4, 5) # save at the END of each; the higher LR/batch peaks earlier, so grab ep2 too
LR           = 2e-4         # raised with the 2x effective batch (32) — standard QLoRA LR, converges faster
SEED         = 7

# "BEST MODEL FAST" recipe — apples-to-apples comparability with cosmos intentionally dropped (user
# wants the best model quickly, not a matched A/B). Per-device batch auto-scales to the GPU AND the
# effective batch is 32 (was 16): fewer, bigger optimizer steps that suit the H100, paired with the
# raised LR below (2e-4) so the larger batch still converges fast. Effective batch is kept equal across
# GPUs so a run is consistent regardless of which card you land on.
import torch
_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
if   _vram >= 90: BS, GA = 16, 2    # G4 Blackwell-96 (RTX PRO 6000) — biggest micro-batch (eff 32)
elif _vram >= 70: BS, GA = 8, 4     # H100-80 / A100-80                          (eff 32)
elif _vram >= 38: BS, GA = 4, 8     # A100-40                                    (eff 32)
else:             BS, GA = 2, 16    # T4-16 / L4-24                              (eff 32)
print(f"GPU ~{_vram:.0f} GB  ->  per-device batch {BS} x grad-accum {GA}  =  effective {BS*GA}")

# Prompts for the post-train voice probe (base = transcript continuation, NOT instructions).
SAMPLE_PROMPTS = [
    "Bugün sizlere anlatmak istediğim bir şey var. ",
    "Bakın şimdi, asıl mesele şu: ",
    "Arkadaşlar, ",
]
"""))

CELLS.append(md("## 5. Helpers — pack (per tokenizer), VRAM cleanup, voice probe"))
CELLS.append(code(r"""
import gc, time, json
from datasets import load_dataset, Dataset
from transformers import TrainerCallback

class EpochAdapterSaver(TrainerCallback):
    # Save the LoRA adapter at the end of each epoch in SAVE_EPOCHS, so we can compare 3 vs 4 vs 5
    # epochs per model after the run (under-baked vs over-cooked) instead of guessing one number.
    def __init__(self, name, tokenizer):
        self.name, self.tokenizer, self.saved = name, tokenizer, {}
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        ep = round(state.epoch)
        if ep in SAVE_EPOCHS:
            path = f"{OUTPUT_ROOT}/{self.name}_lora_ep{ep}"
            model.save_pretrained(path); self.tokenizer.save_pretrained(path)
            self.saved[ep] = path
            print(f"  [checkpoint] {self.name} epoch {ep} -> {path}")

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM free {free/1e9:.1f} / {total/1e9:.1f} GB")

def pack(path, tokenizer):
    # Tokenize each doc, append EOS as the doc boundary, concat, slice into MAX_SEQ_LEN blocks.
    eos = tokenizer.eos_token_id
    assert eos is not None, "tokenizer has no EOS — cannot honor the stop-signal contract"
    docs = load_dataset("json", data_files=path, split="train")
    ids = []
    for ex in docs:
        ids += tokenizer(ex["text"], add_special_tokens=False)["input_ids"] + [eos]
    n = (len(ids) // MAX_SEQ_LEN) * MAX_SEQ_LEN          # drop ragged tail (< 1 block)
    blocks = [ids[i:i + MAX_SEQ_LEN] for i in range(0, n, MAX_SEQ_LEN)]
    ds = Dataset.from_dict({"input_ids": blocks, "labels": [b[:] for b in blocks]})
    # guard against the old packing=True truncation bug + confirm EOS is embedded
    assert len(blocks) > 3 * len(docs), f"only {len(blocks)} blocks from {len(docs)} docs — tokens truncated"
    assert any(eos in b for b in blocks), "EOS not embedded — stop-signal contract broken"
    return len(docs), len(ids), ds

def voice_probe(model, tokenizer):
    # Generate from each prompt; report the text AND whether the model stopped on its own (EOS).
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    samples = []
    for p in SAMPLE_PROMPTS:
        inp = tokenizer(p, return_tensors="pt").to("cuda")
        cap = 200
        out = model.generate(**inp, max_new_tokens=cap, do_sample=True, temperature=0.8,
                             top_p=0.9, repetition_penalty=1.1,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        new = out.shape[1] - inp["input_ids"].shape[1]
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        samples.append({"prompt": p, "text": text, "stopped_on_eos": bool(new < cap), "new_tokens": int(new)})
    return samples
"""))

CELLS.append(md(r"""
## 6. Train all three  ·  *unattended — Run-all reaches here and just works*

Each model: load 4-bit → attach LoRA (assert frozen embeddings) → pack with its own tokenizer →
train → **save adapter to Drive** → voice-probe → free VRAM. Failures are caught per model so the
rest still run.
"""))
CELLS.append(code(r"""
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments, default_data_collator

def train_one(cfg):
    name, model_name = cfg["name"], cfg["model_name"]
    print("\n" + "=" * 72 + f"\nTRAINING  {name}  <-  {model_name}\n" + "=" * 72)
    # idempotent guard: if this model's adapters are already on Drive, skip it. Makes a re-run after a
    # disconnect cheap (finished models are not retrained) and lets this same notebook fill in gemma/llama
    # without touching the already-done cosmos.
    have = {ep: f"{OUTPUT_ROOT}/{name}_lora_ep{ep}" for ep in SAVE_EPOCHS
            if os.path.isdir(f"{OUTPUT_ROOT}/{name}_lora_ep{ep}")}
    if max(SAVE_EPOCHS) in have:
        print(f"SKIP {name}: already trained (adapters on Drive: {sorted(have)})")
        return {"status": "skipped", "minutes": 0.0, "adapters": have,
                "train_loss": [], "eval_loss": [], "samples": []}
    t0 = time.time()
    model = trainer = tokenizer = None
    try:
        free_memory()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
        print("eos:", repr(tokenizer.eos_token), tokenizer.eos_token_id)

        model = FastLanguageModel.get_peft_model(
            model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
            target_modules=TARGETS, use_gradient_checkpointing="unsloth",
            random_state=SEED, use_rslora=False, loftq_config=None)
        # anti-forgetting lever must hold: embeddings + head frozen
        for n_, p_ in model.named_parameters():
            if ("embed_tokens" in n_ or "lm_head" in n_) and p_.requires_grad:
                raise RuntimeError(f"embeddings/head trainable ({n_}) — must be frozen")
        model.print_trainable_parameters()

        n_tr, tok_tr, train_ds = pack(TRAIN, tokenizer)
        n_va, tok_va, val_ds = pack(VAL, tokenizer)
        print(f"train {n_tr} docs -> {tok_tr:,} tok -> {len(train_ds)} blocks | "
              f"val {n_va} docs -> {tok_va:,} tok -> {len(val_ds)} blocks")

        # eval each epoch for the loss curve; the EpochAdapterSaver writes the ep3/4/5 adapters to
        # Drive. We run the FULL 5 epochs (no early-stop) precisely so all three checkpoints exist.
        args = TrainingArguments(
            output_dir=f"/content/outputs_{name}",
            per_device_train_batch_size=BS, gradient_accumulation_steps=GA,
            # Gemma-2's 256k-token vocab makes the eval LOGITS tensor enormous; the Trainer was gathering
            # it and upcasting to fp32 (accelerate _convert_to_fp32 tried to alloc 15.6 GiB) -> OOM at the
            # epoch-1 eval. prediction_loss_only=True makes eval compute ONLY the loss and never materialize
            # logits, which keeps the val-loss curve but kills the OOM. bs=1 is extra insurance. (No-op for llama.)
            per_device_eval_batch_size=1, prediction_loss_only=True,
            num_train_epochs=EPOCHS, learning_rate=LR, lr_scheduler_type="cosine",
            warmup_ratio=0.05, weight_decay=0.01, optim="adamw_8bit",
            logging_steps=10, eval_strategy="epoch", save_strategy="no",
            seed=SEED, report_to="none",
            fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported())

        saver = EpochAdapterSaver(name, tokenizer)
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                          data_collator=default_data_collator, callbacks=[saver])
        trainer.train()

        # safety net: ensure the final epoch's adapter exists even if the callback epoch-rounding missed
        if EPOCHS not in saver.saved:
            p = f"{OUTPUT_ROOT}/{name}_lora_ep{EPOCHS}"
            model.save_pretrained(p); tokenizer.save_pretrained(p); saver.saved[EPOCHS] = p
        print("saved adapters:", saver.saved)

        hist = trainer.state.log_history
        tl = [round(h["loss"], 3) for h in hist if "loss" in h]
        el = [round(h["eval_loss"], 3) for h in hist if "eval_loss" in h]
        samples = voice_probe(model, tokenizer)  # probe the final (epoch-5) model
        res = {"status": "ok", "minutes": round((time.time() - t0) / 60, 1),
               "adapters": saver.saved, "train_loss": tl, "eval_loss": el, "samples": samples}
        if tl and tl[-1] < 0.2:
            res["warn"] = "train loss < 0.2 — possible overfit; scale the adapter down at inference"
    except Exception as e:
        import traceback; traceback.print_exc()
        res = {"status": "FAILED", "minutes": round((time.time() - t0) / 60, 1), "error": str(e)}
    finally:
        del trainer, model, tokenizer
        free_memory()
    print(f"{name}: {res['status']}  ({res['minutes']} min)")
    return res

results = {}
for cfg in MODELS:
    results[cfg["name"]] = train_one(cfg)
    # persist a running summary after each model (survives a later disconnect)
    try:
        with open(f"{OUTPUT_ROOT}/_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
print("\nALL DONE:", {k: v["status"] for k, v in results.items()})
"""))

CELLS.append(md(r"""
## 7. Results — 3 models × {3,4,5} epochs (loss curve + voice + does it stop?)

The voice probe below is the **epoch-5** model of each. The **epoch-3 and epoch-4 adapters are saved
on Drive too** — use the helper in cell 7b to listen to those doses and find where each model peaks
before it starts over-cooking (watch `eval_loss`: the epoch where it turns back up is your ceiling).
"""))
CELLS.append(code(r"""
for name, r in results.items():
    print("=" * 72)
    print(f"{name}  [{r['status']}]  {r.get('minutes','?')} min")
    if r["status"] == "skipped":
        print(f"  skipped — adapters already on Drive: {sorted(r.get('adapters', {}))}"); continue
    if r["status"] != "ok":
        print("  error:", r.get("error")); continue
    print(f"  adapters (epoch -> path): {r['adapters']}")
    print(f"  train_loss (per log step): {r['train_loss']}")
    print(f"  eval_loss  (per epoch)   : {r['eval_loss']}   <- lowest epoch = best dose")
    if r.get("warn"): print("  WARN:", r["warn"])
    for s in r["samples"]:
        stop = "stopped on EOS ✓" if s["stopped_on_eos"] else f"hit {s['new_tokens']}-tok cap (no EOS)"
        print(f"  --- [epoch5] prompt: {s['prompt']!r}  [{stop}]")
        print("     " + s["text"].replace("\n", "\n     "))
print("\nAll adapters are on Drive under", OUTPUT_ROOT)
"""))
CELLS.append(md("### 7b. (optional) Listen to a specific saved adapter — any model, any epoch"))
CELLS.append(code(r"""
# Compare doses: load any saved {name}_lora_ep{N} and voice-probe it. Run per adapter you want to hear.
def probe_saved(adapter_dir):
    free_memory()
    from unsloth import FastLanguageModel
    m, tok = FastLanguageModel.from_pretrained(model_name=adapter_dir,
                                               max_seq_length=MAX_SEQ_LEN, load_in_4bit=True)
    for s in voice_probe(m, tok):
        stop = "stopped on EOS ✓" if s["stopped_on_eos"] else f"hit cap ({s['new_tokens']} tok)"
        print(f"--- {s['prompt']!r}  [{stop}]\n{s['text']}\n")
    del m, tok; free_memory()

# probe_saved(f"{OUTPUT_ROOT}/llama-3.1-8b_lora_ep3")
# probe_saved(f"{OUTPUT_ROOT}/llama-3.1-8b_lora_ep4")
# probe_saved(f"{OUTPUT_ROOT}/cosmos-turkish-8b_lora_ep5")
"""))

CELLS.append(md(r"""
## 8. Evaluation generation — all 9 adapters on a fixed prompt set

Generates each saved adapter's continuation on a fixed set of transcript-style prompts (6 taken from
**held-out real him** + 4 open style probes), checks whether it **stops on its own (EOS)** and whether
it **parrots** training text, and writes `speaker_eval_outputs.json` to Drive. Commit/paste that file and
Claude scores all 9 against `data/voice_profile.md` (ranked base × dose). Run after Section 6.
"""))
CELLS.append(code(r"""
import json, glob
from unsloth import FastLanguageModel

# prompts: prefer the committed file; fall back to an inline set so this works even if only train/val
# were uploaded.
pp = _find("eval_prompts.json")
if pp:
    PROMPTS = json.loads(open(pp, encoding="utf-8").read())
else:
    PROMPTS = [
        {"id": "probe_politics", "type": "open", "prompt": "Şimdi bakın, CHP meselesine gelelim. "},
        {"id": "probe_history",  "type": "open", "prompt": "Öcalan o dönemde ne yapıyordu biliyor musunuz? "},
        {"id": "probe_reaction", "type": "open", "prompt": "Ya bu ne abi ya, "},
        {"id": "probe_geopol",   "type": "open", "prompt": "Amerika bu işin neresinde derseniz, "},
    ]
print(f"{len(PROMPTS)} eval prompts")

# train corpus (one string) for the parroting check
TRAIN_BLOB = " ".join(json.loads(l)["text"] for l in open(TRAIN, encoding="utf-8") if l.strip())

def parrots(text, n=60, step=20):
    t = " ".join(text.split())
    for i in range(0, max(1, len(t) - n), step):
        w = t[i:i + n]
        if len(w) == n and w in TRAIN_BLOB:
            return w
    return None

# every saved adapter from this run: {OUTPUT_ROOT}/{name}_lora_ep{N}
adapters = sorted(glob.glob(f"{OUTPUT_ROOT}/*_lora_ep*"))
print("adapters to eval:", [a.split('/')[-1] for a in adapters])

eval_rows = []
for adir in adapters:
    tag = adir.split("/")[-1]
    print("\n---", tag)
    try:
        free_memory()
        m, tok = FastLanguageModel.from_pretrained(model_name=adir, max_seq_length=MAX_SEQ_LEN,
                                                   load_in_4bit=True)
        FastLanguageModel.for_inference(m)
        for p in PROMPTS:
            inp = tok(p["prompt"], return_tensors="pt").to("cuda")
            cap = 220
            out = m.generate(**inp, max_new_tokens=cap, do_sample=True, temperature=0.7, top_p=0.9,
                             repetition_penalty=1.1, eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.eos_token_id)
            new = out.shape[1] - inp["input_ids"].shape[1]
            full = tok.decode(out[0], skip_special_tokens=True)
            cont = full[len(p["prompt"]):].strip()
            eval_rows.append({
                "adapter": tag, "prompt_id": p["id"], "type": p.get("type"),
                "prompt": p["prompt"], "continuation": cont,
                "stopped_on_eos": bool(new < cap), "new_tokens": int(new),
                "parrot_span": parrots(cont), "reference_real": p.get("reference_real"),
            })
        del m, tok
    except Exception as e:
        import traceback; traceback.print_exc()
        eval_rows.append({"adapter": tag, "error": str(e)})
    finally:
        free_memory()

out_path = f"{OUTPUT_ROOT}/speaker_eval_outputs.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(eval_rows, f, ensure_ascii=False, indent=2)
print("\nWROTE", out_path, f"({len(eval_rows)} rows)")
print("Commit/share this file -> Claude scores all 9 adapters against data/voice_profile.md.")
"""))

CELLS.append(md(r"""
## 9. Export ready-to-run Ollama models  ·  *runs in Run-all — no second notebook*

Turns the trained adapters into usable models on Drive, two ways:
- **llama → light adapter GGUF** (~150 MB). Its base *is* in Ollama's registry, so locally you
  `ollama pull llama3.1:8b-text-q4_K_M` once and the Modelfile layers the adapter on top (the pilot path).
- **gemma → full merged GGUF at Q6_K** (~6.6 GB, self-contained). Its base is NOT in the registry, so it
  must be merged; **Q6** (not Q4) keeps the Turkish sharper.

Modelfiles ship with the coherence-tuned sampling we found (temp 0.6 / top_p 0.85 / repeat 1.15) and raw
passthrough template. Edit `EXPORT` to pick models/epochs. Failures are isolated — training already
finished and the adapters are safe regardless of what happens here.
"""))
CELLS.append(code(r"""
import os, glob, shutil, subprocess
def _sh(c): print("$", c); subprocess.run(c, shell=True, check=True)

# what to turn into ready models (epochs must be ones you saved in SAVE_EPOCHS) ---------------------
EXPORT = {
    "llama-3.1-8b": {"mode": "adapter", "epochs": [3, 4]},                  # tiny adapter GGUFs
    "gemma-2-9b":   {"mode": "merge",   "epochs": [3, 4], "quant": "Q6_K"}, # ~6.6 GB self-contained each
}
LLAMA_BASE_ID     = "unsloth/Meta-Llama-3.1-8B"     # 16-bit CONFIG for the converter (never the 4-bit id)
LLAMA_OLLAMA_BASE = "llama3.1:8b-text-q4_K_M"       # `ollama pull` this locally; the Modelfile FROMs it

# shared Modelfile pieces (no literal triple-quote in source -> '"'*3)
_TPL    = "TEMPLATE " + '"'*3 + "{{ .Prompt }}" + '"'*3 + "\n"
_PARAMS = ("PARAMETER temperature 0.6\nPARAMETER top_p 0.85\nPARAMETER repeat_penalty 1.15\n"
           "PARAMETER num_predict 256\n")
_STOPS  = ('PARAMETER stop "<|end_of_text|>"\nPARAMETER stop "<|eot_id|>"\n'
           'PARAMETER stop "<eos>"\nPARAMETER stop "<end_of_turn>"\n')

# llama.cpp once. We use the DETERMINISTIC path (merge -> convert_hf_to_gguf -> llama-quantize), NOT
# Unsloth's save_pretrained_gguf — its auto-build of llama.cpp is the single most-reported GGUF failure.
LCPP = "/content/llama.cpp"
_need_merge = any(c["mode"] == "merge" for c in EXPORT.values())
if not os.path.isdir(LCPP):
    _sh(f"git clone -q https://github.com/ggml-org/llama.cpp {LCPP}")
    _sh("pip install -q gguf sentencepiece protobuf")          # light deps; don't disturb Unsloth's torch
    if _need_merge:                                            # build the quantizer (CPU build, ~3-4 min)
        _sh(f"cmake -S {LCPP} -B {LCPP}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /content/_lcpp.log 2>&1")
        _sh(f"cmake --build {LCPP}/build -j --target llama-quantize >> /content/_lcpp.log 2>&1")

def _quantize_bin():
    for p in [f"{LCPP}/build/bin/llama-quantize", f"{LCPP}/build/llama-quantize"]:
        if os.path.exists(p): return p
    hits = glob.glob(f"{LCPP}/**/llama-quantize", recursive=True)
    return hits[0] if hits else None

made = []
for name, cfg in EXPORT.items():
    for ep in cfg["epochs"]:
        adir = f"{OUTPUT_ROOT}/{name}_lora_ep{ep}"
        if not os.path.isdir(adir):
            print("skip (no adapter on Drive):", adir); continue
        try:
            if cfg["mode"] == "adapter":
                out = f"{OUTPUT_ROOT}/{name}-ep{ep}-adapter-f16.gguf"
                _sh(f'python {LCPP}/convert_lora_to_gguf.py --base-model-id {LLAMA_BASE_ID} '
                    f'--outtype f16 --outfile "{out}" "{adir}"')
                with open(f"{OUTPUT_ROOT}/Modelfile-{name}-ep{ep}", "w") as f:
                    f.write("FROM " + LLAMA_OLLAMA_BASE + "\nADAPTER ./" + os.path.basename(out) + "\n"
                            + _TPL + _PARAMS + _STOPS)
                made.append(os.path.basename(out))
            else:
                # gemma: deterministic merge -> convert_hf_to_gguf -> llama-quantize (no Unsloth auto-build)
                from unsloth import FastLanguageModel
                free_memory()
                m, tok = FastLanguageModel.from_pretrained(model_name=adir, max_seq_length=MAX_SEQ_LEN,
                                                           dtype=None, load_in_4bit=True)
                merged = f"/content/merged_{name}_{ep}"
                m.save_pretrained_merged(merged, tok, save_method="merged_16bit")
                del m, tok; free_memory()
                q = cfg.get("quant", "Q6_K")
                f16 = f"/content/{name}_{ep}_f16.gguf"
                _sh(f'python {LCPP}/convert_hf_to_gguf.py "{merged}" --outfile "{f16}" --outtype f16')
                gguf = f"{name}-ep{ep}-{q.lower()}.gguf"
                _sh(f'"{_quantize_bin()}" "{f16}" "{OUTPUT_ROOT}/{gguf}" {q}')
                os.remove(f16); shutil.rmtree(merged, ignore_errors=True)
                with open(f"{OUTPUT_ROOT}/Modelfile-{name}-ep{ep}", "w") as f:
                    f.write("FROM ./" + gguf + "\n" + _TPL + _PARAMS + _STOPS)
                made.append(gguf)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  EXPORT FAILED for {name} ep{ep}: {e}  (adapter is still safe on Drive)")

print("\nExported to Drive:", made)
print("Locally:")
print("  gemma:  ollama create speaker-gemma-ep3 -f Modelfile-gemma-2-9b-ep3   &&  ollama run speaker-gemma-ep3")
print("  llama:  ollama pull llama3.1:8b-text-q4_K_M")
print("          ollama create speaker-llama-ep3 -f Modelfile-llama-3.1-8b-ep3 &&  ollama run speaker-llama-ep3")
"""))

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "qlora_train_multi.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print("wrote", out, f"({len(CELLS)} cells)")
