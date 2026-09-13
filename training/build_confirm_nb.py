"""Generate training/qlora_confirm.ipynb — the ONE cheap "is-it-the-recipe?" experiment.

Background (2026-06-27 deep research, see memory/turkish_collapse_diagnosis.md):
the "Turkish generation collapse" (ramble / repeat / hallucinate) is almost certainly
RECIPE-driven over-fitting, NOT a Turkish-capability deficit. Smoking gun: cosmos-turkish-8b
(already Turkish-pretrained) collapsed too, under the same aggressive recipe.

This notebook isolates that single variable. SAME base (cosmos-turkish-8b), SAME dataset,
SAME everything — EXCEPT the recipe goes gentle:

      old (collapsed)        ->   confirm (gentle)
  r       64                 ->   32
  alpha   128 (=2r)          ->   64  (=2r, ratio kept)
  LR      2e-4               ->   1e-4
  epochs  5                  ->   2   (adapters saved at ep1 AND ep2)
  embeddings  frozen         ->   frozen   (unchanged)
  eff batch   32             ->   32       (unchanged — keep batch out of the variable set)

If the gentle cosmos stays coherent / stops rambling -> diagnosis CONFIRMED -> we then invest
in the clean dataset + final base choice + (maybe) Turkish replay. If it STILL collapses ->
diagnosis wrong, escalate. Output lands in a SEPARATE Drive folder so the old cosmos is kept
for a head-to-head, and Section 7/8 probe OLD-aggressive vs NEW-gentle on the same prompts.

Re-run after editing CELLS:  python training/build_confirm_nb.py
"""
from __future__ import annotations
import json
from pathlib import Path

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
def code(text): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}

CELLS = []

CELLS.append(md(r"""
# Speaker voice-clone — **CONFIRM test: is the collapse the recipe?** (Unsloth)

One cheap run to settle the question we've been going in circles on. Deep research
(2026-06-27) says the "Turkish generation collapse" is **recipe-driven over-fitting, not a
Turkish-capability deficit** — and the smoking gun is that **cosmos-turkish-8b collapsed too**,
even though it's already continued-pretrained on Turkish.

So we change exactly ONE thing — the recipe — and keep the base and data identical:

| | old run (collapsed) | **this confirm run (gentle)** |
|---|---|---|
| base | `ytu-ce-cosmos/Turkish-Llama-8b-v0.1` | **same** |
| dataset | `train.jsonl` / `val.jsonl` (432/47) | **same** |
| LoRA r | 64 | **32** |
| alpha | 128 (=2r) | **64** (=2r, ratio kept) |
| learning rate | 2e-4 | **1e-4** |
| epochs | 5 | **2** (adapters saved at ep1 **and** ep2) |
| embeddings | frozen | frozen (unchanged) |
| effective batch | 32 | 32 (unchanged — kept out of the variable set) |

**Read the result like this:** the win condition is **coherent Turkish that stops on its own
and does NOT ramble/repeat/hallucinate** — voice quality is secondary for now. Section 7 plays
the OLD aggressive cosmos and the NEW gentle cosmos on the *same* prompts so you can hear the
difference directly; Section 8 writes `speaker_confirm_eval_outputs.json` with BOTH for Claude to score.

**EOS contract (unchanged):** the data file carries no literal EOS; this notebook appends
`tokenizer.eos_token` between docs so the model learns to emit a real stop token at doc
boundaries. The probe reports whether each model actually stops on its own.

**Run in the Colab *web* UI (browser), not the VS Code bridge** (the bridge drops on long jobs).
Any GPU works for *training* — this is ONE 8B model for 2 epochs, so even **T4 ≈ 2.5–3.5 h**,
A100 ≈ 1 h, H100 ≈ 30–45 min. Training needs no High-RAM toggle. The optional **§9 GGUF export**
(so you can run him locally) DOES an fp16 merge (~16 GB) — on a free T4 with standard RAM that can
OOM; prefer an **A100** or a **High-RAM** runtime if you want the GGUF in the same run.

**What you do:** run the 3 interactive cells at the top once (HF login → Drive → data), then
**Runtime ▸ Run all** and walk away. You get, all on Drive under `speaker_confirm/`: the gentle
adapters (ep1+ep2), `speaker_confirm_eval_outputs.json` (for Claude to score), and a **Q4_K_M GGUF +
Modelfile** ready for local Ollama on the 8 GB GPU.
""".lstrip("\n")))

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
"""))

CELLS.append(md(r"""
## 2. Auth + persistence  ·  **interactive — do this once**

- **HF login**: the Cosmos base lives behind a license click. Accept it on huggingface.co with
  your account first (one-time), then paste a *read* token here.
- **Drive mount**: where the confirm adapters land. We use a SEPARATE folder (`speaker_confirm`)
  so the old cosmos under `speaker_models` is kept for the head-to-head comparison.
"""))
CELLS.append(code(r"""
import torch, os
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
      "| bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)

from huggingface_hub import login
login()  # paste a READ token; accept the cosmos license on its HF page first

USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = "/content/drive/MyDrive/speaker_confirm"     # NEW folder — keeps the old run intact
    OLD_ROOT    = "/content/drive/MyDrive/speaker_models"      # where the old aggressive cosmos lives
else:
    OUTPUT_ROOT = "/content/speaker_confirm"
    OLD_ROOT    = "/content/speaker_models"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print("confirm adapters ->", OUTPUT_ROOT)
print("old run (for compare) ->", OLD_ROOT, "(exists)" if os.path.isdir(OLD_ROOT) else "(not found — compare cells will skip)")
"""))

CELLS.append(md(r"""
## 3. Data  ·  **interactive — upload or clone train/val**

Right-click `data/dataset/train.jsonl` + `val.jsonl` in VS Code → *Upload to Colab*, **or**
uncomment the git-clone block (needs a GitHub PAT). Same files as the old run — do not re-clean.
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

# --- git-clone alternative ---
# import getpass; PAT = getpass.getpass("GitHub PAT: ")
# !git clone --depth 1 https://{PAT}@github.com/mehmettahacumurcu/youtuber-clone.git _repo
# TRAIN, VAL = "_repo/data/dataset/train.jsonl", "_repo/data/dataset/val.jsonl"

assert TRAIN and VAL, "train.jsonl / val.jsonl not found — upload them or use the git-clone block above."
print("train:", TRAIN, os.path.getsize(TRAIN), "B | val:", VAL, os.path.getsize(VAL), "B")
"""))

CELLS.append(md(r"""
## 4. Config — cosmos only, **gentle** recipe

Only the recipe knobs differ from the old run (r, alpha, LR, epochs). Everything else — base,
data, frozen embeddings, targets, scheduler, warmup, weight decay, optimizer, seed, batch — is
held identical so a clean result is attributable to the recipe alone.
"""))
CELLS.append(code(r"""
# Single model — the one that collapsed despite being already-Turkish (that's the whole point).
MODELS = [
    {"name": "cosmos-confirm", "model_name": "ytu-ce-cosmos/Turkish-Llama-8b-v0.1"},
]

# --- GENTLE QLoRA hyperparameters (the ONLY change vs the collapsed run) ---
MAX_SEQ_LEN  = 2048
LORA_R       = 32          # was 64  -> lower rank = fewer "intruder dimensions" (arxiv 2410.21228)
LORA_ALPHA   = 64          # was 128 -> kept at 2*r (rank-stabilized ratio)
LORA_DROPOUT = 0.05
TARGETS      = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EPOCHS       = 2           # was 5   -> 4.4M tokens over-bakes well before 5 epochs
SAVE_EPOCHS  = (1, 2)      # save BOTH so we can see the dose (is even 1 epoch enough?)
LR           = 1e-4        # was 2e-4 -> high LR raises intruder dimensions / forgetting; halve it
SEED         = 7

# effective batch held at 32 to MATCH the old run (keep batch out of the variable set).
import torch
_vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
if   _vram >= 90: BS, GA = 16, 2
elif _vram >= 70: BS, GA = 8, 4
elif _vram >= 38: BS, GA = 4, 8
else:             BS, GA = 2, 16
print(f"GPU ~{_vram:.0f} GB  ->  per-device batch {BS} x grad-accum {GA}  =  effective {BS*GA}")

# transcript-continuation probes (base model = continuation, NOT instructions)
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
    eos = tokenizer.eos_token_id
    assert eos is not None, "tokenizer has no EOS — cannot honor the stop-signal contract"
    docs = load_dataset("json", data_files=path, split="train")
    ids = []
    for ex in docs:
        ids += tokenizer(ex["text"], add_special_tokens=False)["input_ids"] + [eos]
    n = (len(ids) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    blocks = [ids[i:i + MAX_SEQ_LEN] for i in range(0, n, MAX_SEQ_LEN)]
    ds = Dataset.from_dict({"input_ids": blocks, "labels": [b[:] for b in blocks]})
    assert len(blocks) > 3 * len(docs), f"only {len(blocks)} blocks from {len(docs)} docs — tokens truncated"
    assert any(eos in b for b in blocks), "EOS not embedded — stop-signal contract broken"
    return len(docs), len(ids), ds

def voice_probe(model, tokenizer):
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
## 6. Train  ·  *unattended — Run-all reaches here and just works*

Load 4-bit → attach gentle LoRA (assert frozen embeddings) → pack → train 2 epochs (save ep1+ep2)
→ voice-probe → free VRAM.
"""))
CELLS.append(code(r"""
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments, default_data_collator

def train_one(cfg):
    name, model_name = cfg["name"], cfg["model_name"]
    print("\n" + "=" * 72 + f"\nTRAINING  {name}  <-  {model_name}\n" + "=" * 72)
    have = {ep: f"{OUTPUT_ROOT}/{name}_lora_ep{ep}" for ep in SAVE_EPOCHS
            if os.path.isdir(f"{OUTPUT_ROOT}/{name}_lora_ep{ep}")}
    if max(SAVE_EPOCHS) in have:
        print(f"SKIP {name}: already trained (adapters: {sorted(have)})")
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
        for n_, p_ in model.named_parameters():
            if ("embed_tokens" in n_ or "lm_head" in n_) and p_.requires_grad:
                raise RuntimeError(f"embeddings/head trainable ({n_}) — must be frozen")
        model.print_trainable_parameters()

        n_tr, tok_tr, train_ds = pack(TRAIN, tokenizer)
        n_va, tok_va, val_ds = pack(VAL, tokenizer)
        print(f"train {n_tr} docs -> {tok_tr:,} tok -> {len(train_ds)} blocks | "
              f"val {n_va} docs -> {tok_va:,} tok -> {len(val_ds)} blocks")

        args = TrainingArguments(
            output_dir=f"/content/outputs_{name}",
            per_device_train_batch_size=BS, gradient_accumulation_steps=GA,
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

        if EPOCHS not in saver.saved:
            p = f"{OUTPUT_ROOT}/{name}_lora_ep{EPOCHS}"
            model.save_pretrained(p); tokenizer.save_pretrained(p); saver.saved[EPOCHS] = p
        print("saved adapters:", saver.saved)

        hist = trainer.state.log_history
        tl = [round(h["loss"], 3) for h in hist if "loss" in h]
        el = [round(h["eval_loss"], 3) for h in hist if "eval_loss" in h]
        samples = voice_probe(model, tokenizer)
        res = {"status": "ok", "minutes": round((time.time() - t0) / 60, 1),
               "adapters": saver.saved, "train_loss": tl, "eval_loss": el, "samples": samples}
        if tl and tl[-1] < 0.2:
            res["warn"] = "train loss < 0.2 — still over-fitting; drop LR or epochs further"
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
    try:
        with open(f"{OUTPUT_ROOT}/_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
print("\nDONE:", {k: v["status"] for k, v in results.items()})
"""))

CELLS.append(md(r"""
## 7. Result — gentle cosmos: loss curve, voice, **does it stop?**

The win condition is **coherent Turkish that stops on its own (EOS ✓) and does NOT ramble/repeat**.
`eval_loss` per epoch: if ep2 > ep1 it's already over-baking at 2 epochs (use ep1). A `train loss
< 0.2` warning means it's STILL over-fitting and we drop LR/epochs again.
"""))
CELLS.append(code(r"""
for name, r in results.items():
    print("=" * 72)
    print(f"{name}  [{r['status']}]  {r.get('minutes','?')} min")
    if r["status"] != "ok":
        print("  ", r.get("error") or r.get("adapters")); continue
    print(f"  adapters: {r['adapters']}")
    print(f"  train_loss: {r['train_loss']}")
    print(f"  eval_loss (per epoch): {r['eval_loss']}   <- lowest epoch = best dose")
    if r.get("warn"): print("  WARN:", r["warn"])
    for s in r["samples"]:
        stop = "stopped on EOS ✓" if s["stopped_on_eos"] else f"hit {s['new_tokens']}-tok cap (NO EOS — rambling)"
        print(f"  --- prompt: {s['prompt']!r}  [{stop}]")
        print("     " + s["text"].replace("\n", "\n     "))
print("\nadapters on Drive under", OUTPUT_ROOT)
"""))

CELLS.append(md(r"""
## 7b. **Head-to-head: OLD aggressive cosmos vs NEW gentle cosmos** (the actual answer)

Plays the old run's `cosmos-turkish-8b_lora_ep5` (r64 / 2e-4 / 5ep) and the new gentle
`cosmos-confirm_lora_ep2` on the SAME prompts. If old rambles and new stays coherent, the
diagnosis is confirmed and we move on. Skips gracefully if the old adapter isn't on Drive.
"""))
CELLS.append(code(r"""
def probe_saved(adapter_dir, label):
    if not os.path.isdir(adapter_dir):
        print(f"[skip] {label}: not found at {adapter_dir}"); return
    print("\n" + "#" * 72 + f"\n# {label}\n# {adapter_dir}\n" + "#" * 72)
    free_memory()
    from unsloth import FastLanguageModel
    m, tok = FastLanguageModel.from_pretrained(model_name=adapter_dir,
                                               max_seq_length=MAX_SEQ_LEN, load_in_4bit=True)
    for s in voice_probe(m, tok):
        stop = "stopped on EOS ✓" if s["stopped_on_eos"] else f"hit cap ({s['new_tokens']} tok — rambling)"
        print(f"--- {s['prompt']!r}  [{stop}]\n{s['text']}\n")
    del m, tok; free_memory()

probe_saved(f"{OLD_ROOT}/cosmos-turkish-8b_lora_ep5", "OLD — aggressive (r64 / 2e-4 / 5ep)")
probe_saved(f"{OUTPUT_ROOT}/cosmos-confirm_lora_ep2", "NEW — gentle (r32 / 1e-4 / 2ep)")
# also hear the 1-epoch dose:
# probe_saved(f"{OUTPUT_ROOT}/cosmos-confirm_lora_ep1", "NEW — gentle, 1 epoch")
"""))

CELLS.append(md(r"""
## 8. Eval generation — OLD + NEW on the fixed prompt set → `speaker_confirm_eval_outputs.json`

Generates both the old aggressive cosmos and the new gentle cosmos on the committed
`eval_prompts.json` (held-out real-him prompts + open probes), flags EOS-stop + parroting, and
writes one JSON with BOTH so Claude can score them head-to-head against `data/voice_profile.md`.
"""))
CELLS.append(code(r"""
import json, glob
from unsloth import FastLanguageModel

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

TRAIN_BLOB = " ".join(json.loads(l)["text"] for l in open(TRAIN, encoding="utf-8") if l.strip())
def parrots(text, n=60, step=20):
    t = " ".join(text.split())
    for i in range(0, max(1, len(t) - n), step):
        w = t[i:i + n]
        if len(w) == n and w in TRAIN_BLOB:
            return w
    return None

# the two we want to compare head-to-head (old aggressive vs new gentle), plus the ep1 dose if present
adapters = [a for a in [
    f"{OLD_ROOT}/cosmos-turkish-8b_lora_ep5",
    f"{OUTPUT_ROOT}/cosmos-confirm_lora_ep1",
    f"{OUTPUT_ROOT}/cosmos-confirm_lora_ep2",
] if os.path.isdir(a)]
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

out_path = f"{OUTPUT_ROOT}/speaker_confirm_eval_outputs.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(eval_rows, f, ensure_ascii=False, indent=2)
print("\nWROTE", out_path, f"({len(eval_rows)} rows)")
print("Commit/share this file -> Claude scores OLD-aggressive vs NEW-gentle and we decide the next step.")
"""))

CELLS.append(md(r"""
## 9. Export gentle cosmos → **GGUF on Drive** (run him locally on the 8 GB GPU)

You wanted to keep this model and test it locally too. This merges the gentle adapter into the
base, quantizes to **Q4_K_M** (~5 GB — fits an RTX 3060 Ti 8 GB with room for context), and writes
the `.gguf` + a matching **Modelfile** to the same `speaker_confirm/` Drive folder. Download both, then
`ollama create` locally (PowerShell steps in §10).

Defaults to **ep2** (the headline 2-epoch gentle dose). Set `EXPORT_EPOCHS = [1, 2]` to get both
doses as separate GGUFs. **fp16-merge note:** this materializes a 16-bit 8B (~16 GB) to merge the
LoRA — RAM-heavy. On a free **T4 with standard RAM it can OOM**; if so use an **A100** / **High-RAM**
runtime, or run `training/export_cosmos_gguf.ipynb` separately later (it points at the same adapter).
A §9 OOM does NOT lose anything — the adapters and eval JSON are already on Drive from §6–8.
"""))
CELLS.append(code(r"""
import glob, shutil, subprocess
from unsloth import FastLanguageModel

EXPORT_EPOCHS = [2]          # headline gentle dose; set [1, 2] for both doses
QUANT         = "q4_k_m"     # ~5 GB, fits 8 GB; "q5_k_m" for a touch more quality if you have room
EXPORT_NAME   = "cosmos-confirm"

def sh(cmd):
    print("$", cmd); subprocess.run(cmd, shell=True, check=True)

def write_modelfile(epoch, gguf_name):
    # raw passthrough template -> feeds the prompt verbatim (this is a BASE/transcript-completion
    # model; a chat template would mismatch training). Llama-3 stop tokens (cosmos = Turkish-Llama-8b).
    lines = [
        "FROM ./" + gguf_name,
        "TEMPLATE " + '"' * 3 + "{{ .Prompt }}" + '"' * 3,
        "PARAMETER temperature 0.8",
        "PARAMETER top_p 0.9",
        "PARAMETER repeat_penalty 1.1",
        "PARAMETER num_predict 256",
        'PARAMETER stop "<|end_of_text|>"',
        'PARAMETER stop "<|eot_id|>"',
    ]
    path = f"{OUTPUT_ROOT}/Modelfile-{EXPORT_NAME}-ep{epoch}"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path

def export_one(epoch):
    adapter = f"{OUTPUT_ROOT}/{EXPORT_NAME}_lora_ep{epoch}"
    assert os.path.isdir(adapter), f"adapter not found (train §6 first): {adapter}"
    out_gguf  = f"{OUTPUT_ROOT}/{EXPORT_NAME}-ep{epoch}-{QUANT}.gguf"
    gguf_name = os.path.basename(out_gguf)
    print("\n" + "=" * 70 + f"\nEXPORT ep{epoch}\n  adapter: {adapter}\n  -> {out_gguf}\n" + "=" * 70)
    free_memory()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=adapter, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    try:
        # primary: Unsloth one-shot merge + convert + quantize
        tmp = f"/content/gguf_ep{epoch}"
        model.save_pretrained_gguf(tmp, tok, quantization_method=QUANT)
        hits = glob.glob(f"{tmp}/*{QUANT.upper()}*.gguf") or glob.glob(f"{tmp}/*.gguf")
        assert hits, "no .gguf produced by save_pretrained_gguf"
        shutil.copy(hits[0], out_gguf); shutil.rmtree(tmp, ignore_errors=True)
    except Exception as ex:
        print("\n[primary GGUF path failed -> manual llama.cpp fallback]\n", repr(ex))
        merged = f"/content/merged_ep{epoch}"
        model.save_pretrained_merged(merged, tok, save_method="merged_16bit")
        if not os.path.exists("/content/llama.cpp"):
            sh("git clone -q https://github.com/ggerganov/llama.cpp /content/llama.cpp")
            sh("cmake -S /content/llama.cpp -B /content/llama.cpp/build -DGGML_CUDA=OFF "
               "-DLLAMA_CURL=OFF > /content/_cmake.log 2>&1")
            sh("cmake --build /content/llama.cpp/build -j --target llama-quantize >> /content/_cmake.log 2>&1")
        f16 = f"/content/f16_ep{epoch}.gguf"
        sh(f"python /content/llama.cpp/convert_hf_to_gguf.py {merged} --outfile {f16} --outtype f16")
        sh(f'/content/llama.cpp/build/bin/llama-quantize "{f16}" "{out_gguf}" Q4_K_M')
        os.remove(f16); shutil.rmtree(merged, ignore_errors=True)
    mf = write_modelfile(epoch, gguf_name)
    del model, tok; free_memory()
    size_gb = round(os.path.getsize(out_gguf) / 1e9, 2)
    print(f"\nDONE ep{epoch}: {gguf_name} ({size_gb} GB) + {os.path.basename(mf)}")
    return out_gguf, mf

produced = []
for e in EXPORT_EPOCHS:
    try:
        produced.append(export_one(e))
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"\n[ep{e} export FAILED] {ex}\n  -> if OOM: use an A100 / High-RAM runtime, or run "
              f"training/export_cosmos_gguf.ipynb later (adapters are safe on Drive).")

print("\nGGUF on Drive (download the .gguf + its Modelfile into one local folder):")
sh(f'ls -lh "{OUTPUT_ROOT}"/*.gguf "{OUTPUT_ROOT}"/Modelfile-* 2>/dev/null || true')
"""))

CELLS.append(md(r"""
## 10. Run him locally (Ollama, on your RTX 3060 Ti 8 GB)

1. From `MyDrive/speaker_confirm/` download the **`.gguf`** (e.g. `cosmos-confirm-ep2-q4_k_m.gguf`)
   and its matching **`Modelfile-cosmos-confirm-ep2`** into the *same* local folder
   (e.g. `E:\youtuber-clone\models\`).
2. In **PowerShell**, from that folder:
   ```powershell
   ollama create cosmos-confirm-ep2 -f Modelfile-cosmos-confirm-ep2
   ollama run cosmos-confirm-ep2
   ```
3. At the `>>>` prompt give it a transcript-style opener and let him continue:
   ```
   Bakın şimdi, asıl mesele şu:
   ```

For a clean base-completion test (no chat quirks), hit the raw API:
```powershell
curl http://localhost:11434/api/generate -d '{\"model\":\"cosmos-confirm-ep2\",\"prompt\":\"Bakın şimdi, asıl mesele şu:\",\"raw\":true,\"stream\":false}'
```

**Listen for the win condition:** coherent Turkish that **stops on its own**, his tics
(yani/abi/bak/işte), the livestream register, profanity landing naturally — and crucially that it
does NOT ramble/repeat/hallucinate the way the old r64/2e-4/5ep cosmos did. That contrast is the
whole experiment.
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

out = Path(__file__).parent / "qlora_confirm.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print("wrote", out, f"({len(CELLS)} cells)")
