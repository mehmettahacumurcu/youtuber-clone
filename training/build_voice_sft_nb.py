"""Generator for training/qwen_voice_sft.ipynb.

4-RECIPE SWEEP + AUTO-SCORING: soft-touch QLoRA SFT of the abliterated Qwen3-14B (v2) on the full
2,428-row closed-book opinion set (voice_sft_v3_train.jsonl). Trains FOUR recipes on one H100, saves
EVERY epoch adapter to Drive, then SCORES every checkpoint on a fixed probe battery and writes a score
file (JSON + readable Markdown, with the full outputs) to Drive. Send that score file back to pick a winner.

  A gentle    r16/a32  LR1e-4  2ep   <- main recipe (safe style transfer)
  B capacity  r32/a64  LR1e-4  2ep   <- more LoRA room: does r16 bottleneck his 2,147 opinions?
  C slowbake  r16/a32  LR5e-5  3ep   <- low-LR long bake: smoother imprint / better generalization?
  D capslow   r32/a64  LR5e-5  3ep   <- B+C together: big room AND gentle long bake

Data is IDENTICAL across all arms, so any difference is the recipe alone (clean 2x2: capacity x schedule).

Everything else is deliberate:
- Base `huihui-ai/Huihui-Qwen3-14B-abliterated-v2`: stable + clean Turkish, abliterated so his crude
  profanity survives (the SFT teaches it to USE that profanity, which the base resists zero-shot).
- Qwen3 hybrid, trained + inferred THINKING-FREE on plain ChatML; <|im_end|> the learned stop token.
- SFT with loss MASKED to the assistant turn only (train_on_responses_only), embeddings/head FROZEN.
- Closed-book: bare question -> his take (NO RAG context).

Run:  python training/build_voice_sft_nb.py  ->  training/qwen_voice_sft.ipynb
"""
from __future__ import annotations
import json
from pathlib import Path

CELLS = []
def md(src): CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})

md(r"""# Speaker voice SFT — 4-recipe QLoRA sweep + auto-scoring on **H100** (abliterated Qwen3-14B v2)

**Run-all →** trains 4 recipes, saves **every epoch** to Drive, then **scores every checkpoint** on a
fixed probe battery and writes a score file (JSON + readable Markdown w/ full outputs) to Drive.
Send that score file back → we pick which to convert to GGUF and run first.

| arm | r / α | LR | epochs | tests |
|---|---|---|---|---|
| **A `gentle`** | 16 / 32 | 1e-4 | 2 | our main recipe — safe style transfer, low collapse risk |
| **B `capacity`** | 32 / 64 | 1e-4 | 2 | does r16 bottleneck his 2,147 opinions? more adapter room |
| **C `slowbake`** | 16 / 32 | 5e-5 | 3 | low-LR long bake — smoother imprint / better generalization? |
| **D `capslow`** | 32 / 64 | 5e-5 | 3 | B+C together — big room **and** gentle long bake |

- **Data identical across all four** = `voice_sft_v3_train.jsonl` (2,428 rows, closed-book, opinions baked in).
- **Every epoch saved to Drive**: `qwen3_v3sweep_<arm>_ep<N>` (ep1/ep2 for A,B; ep1/ep2/ep3 for C,D → 10 checkpoints).
- **Auto-scoring** (Section 8) reloads each checkpoint, runs 14 fixed probes (his broken topics + voice +
  fact), computes collapse/voice metrics, and writes `speaker_scores_v3sweep.{json,md}` to Drive.
- Hardware: **H100-80GB** (fastest). NOT a 16GB T4. **~2.5–3.5h train + ~20–30 min score.**
- **Robust:** train (Sec 7) saves all adapters FIRST; scoring (Sec 8) and GGUF export (Sec 10) are separate
  re-runnable steps that read adapters back from Drive — a disconnect never costs a trained model.
- **Run in Colab web (browser), not the VS Code extension.**
""")

md("## 1. Install")
code(r"""%%capture
!pip install unsloth
!pip install -q gguf protobuf sentencepiece   # for the GGUF export path""")
code(r"""import unsloth, trl, transformers, peft
print("unsloth", unsloth.__version__, "| trl", trl.__version__,
      "| transformers", transformers.__version__, "| peft", peft.__version__)""")

md(r"""## 2. Auth + Drive + GPU check (interactive, once)""")
code(r"""import os, torch
assert torch.cuda.is_available(), "No GPU — pick H100 (fastest) or A100 in Runtime > Change runtime type."
_name = torch.cuda.get_device_name(0)
_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print("GPU:", _name, f"{_vram:.0f} GB")
assert _vram >= 22, f"Only {_vram:.0f} GB — 14B-4bit SFT needs >=24GB (H100/A100/L4); a 16GB T4 OOMs at load."
if _vram < 70:
    print("  [note] <70GB — if you OOM on an r32 arm, set BS,GA = 1,8 in Section 4.")
from huggingface_hub import login
login()  # paste a READ token
from google.colab import drive
drive.mount("/content/drive")
OUTPUT_ROOT = "/content/drive/MyDrive/speaker_models"
RUN = "v3sweep"   # tags EVERY output so this sweep never clobbers the earlier run2 files on Drive
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print("outputs ->", OUTPUT_ROOT, "| run tag:", RUN)""")

md(r"""## 3. Data — upload `voice_sft_v3_train.jsonl` + `voice_sft_val.jsonl`

Each line: `{"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}`.
Right-click → Upload in the Colab file panel, or uncomment the git-clone block.""")
code(r"""import glob
def _find(name):
    for d in [".", "/content", "/content/drive/MyDrive"]:
        p = os.path.join(d, name)
        if os.path.isfile(p): return p
    hits = glob.glob(f"/content/**/{name}", recursive=True) or glob.glob(f"**/{name}", recursive=True)
    return hits[0] if hits else None
TRAIN_FILE = "voice_sft_v3_train.jsonl"   # full 2,428-row opinion-baked set; swap to "voice_sft_train.jsonl" for the 374-row proof
TRAIN = _find(TRAIN_FILE); VAL = _find("voice_sft_val.jsonl")
# import getpass; PAT=getpass.getpass("GitHub PAT: ")
# !git clone --depth 1 https://{PAT}@github.com/mehmettahacumurcu/youtuber-clone.git _repo
# TRAIN=f"_repo/data/dataset/{TRAIN_FILE}"; VAL="_repo/data/dataset/voice_sft_val.jsonl"
assert TRAIN and VAL, f"{TRAIN_FILE} / voice_sft_val.jsonl not found — upload them (Colab file panel, right-click Upload)."
_n = sum(1 for _l in open(TRAIN, encoding="utf-8") if _l.strip())
assert _n >= 250, f"Only {_n} train rows — expected 2428 (full v3) or 374 (proof); check TRAIN_FILE / re-upload."
print(f"train: {TRAIN} ({_n} rows) | val: {VAL}")""")

md(r"""## 4. Config — the 4 recipes + shared constants (SAVE ALL EPOCHS)""")
code(r"""MODEL_NAME   = "huihui-ai/Huihui-Qwen3-14B-abliterated-v2"   # Qwen3 hybrid base; trained THINKING-FREE
MAX_SEQ_LEN  = 4096
TARGETS      = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]  # all linear
LORA_DROPOUT = 0.05
SEED         = 7
BS, GA       = 2, 4        # eff-batch 8 (H100/A100 fit BS=2 @ seq4096). Drop to 1,8 if you OOM on an r32 arm.

# each arm: name | rank/alpha (alpha=2r) | LR | epochs | SAVE = every epoch (nothing discarded)
RECIPES = [
    {"name":"gentle",   "r":16, "alpha":32, "lr":1e-4, "epochs":2, "save":(1,2)},    # A: main / safe
    {"name":"capacity", "r":32, "alpha":64, "lr":1e-4, "epochs":2, "save":(1,2)},    # B: more LoRA room
    {"name":"slowbake", "r":16, "alpha":32, "lr":5e-5, "epochs":3, "save":(1,2,3)},  # C: low-LR long bake
    {"name":"capslow",  "r":32, "alpha":64, "lr":5e-5, "epochs":3, "save":(1,2,3)},  # D: B+C combined
]

# context-free SYS — MUST match the dataset's system prompt and rag/ask_finetuned.py / say.py.
SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")
print("recipes:", [r["name"] for r in RECIPES], "| every epoch saved to Drive")""")

md(r"""## 5. Helpers — load fresh base, train one recipe (saves every epoch)

`run_recipe` reloads the 4-bit base FRESH per arm (no adapter/optimizer state leaks), attaches that arm's
LoRA, freezes embeddings, trains with loss masked to the assistant turn only, and saves EVERY epoch's
adapter to Drive. Scoring (Section 8) is separate so we score all 10 checkpoints uniformly.""")
code(r"""from datasets import load_dataset
import torch, time, gc

def load_fresh():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    tok = get_chat_template(tok, chat_template="qwen-2.5", map_eos_token=True)  # plain thinking-free ChatML
    tok.truncation_side = "left"   # overflow trims the FRONT, never the answer+<|im_end|>
    return model, tok

def run_recipe(rc):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    model, tok = load_fresh()
    model = FastLanguageModel.get_peft_model(
        model, r=rc["r"], lora_alpha=rc["alpha"], lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=TARGETS, use_gradient_checkpointing="unsloth", random_state=SEED)
    for n_, p_ in model.named_parameters():          # anti-forgetting: embeddings/head must stay frozen
        if ("embed_tokens" in n_ or "lm_head" in n_) and p_.requires_grad:
            raise RuntimeError(f"embeddings trainable ({n_}) — must be frozen")

    def _fmt(ex): return {"text": tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}
    train_ds = load_dataset("json", data_files=TRAIN, split="train").map(_fmt)
    val_ds   = load_dataset("json", data_files=VAL,   split="train").map(_fmt)
    _steps = max(1, len(train_ds)//(BS*GA)) * rc["epochs"]
    print(f"  {rc['name']}: {len(train_ds)} rows x {rc['epochs']}ep / eff-batch {BS*GA} = ~{_steps} steps")

    saved = {}
    class Saver(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kw):
            ep = round(state.epoch)
            if ep in rc["save"]:
                p = f"{OUTPUT_ROOT}/qwen3_{RUN}_{rc['name']}_ep{ep}"
                model.save_pretrained(p); tok.save_pretrained(p); saved[ep] = p
                print(f"  [ckpt] {rc['name']} ep{ep} -> {p}")

    args = SFTConfig(
        output_dir=f"/content/out_{rc['name']}", dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BS, gradient_accumulation_steps=GA,
        num_train_epochs=rc["epochs"], learning_rate=rc["lr"], lr_scheduler_type="cosine", warmup_ratio=0.05,
        weight_decay=0.01, optim="adamw_8bit", logging_steps=10,
        eval_strategy="epoch", per_device_eval_batch_size=1, prediction_loss_only=True,
        save_strategy="no", seed=SEED, report_to="none",
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported())
    try:   # newer TRL renamed SFTTrainer's tokenizer= -> processing_class=
        tr = SFTTrainer(model=model, tokenizer=tok, train_dataset=train_ds, eval_dataset=val_ds, args=args, callbacks=[Saver()])
    except TypeError:
        tr = SFTTrainer(model=model, processing_class=tok, train_dataset=train_ds, eval_dataset=val_ds, args=args, callbacks=[Saver()])
    tr = train_on_responses_only(tr, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    t0 = time.time(); tr.train()
    for ep in rc["save"]:            # safety: ensure every requested epoch got saved
        if ep not in saved and ep <= rc["epochs"]:
            p = f"{OUTPUT_ROOT}/qwen3_{RUN}_{rc['name']}_ep{ep}"; model.save_pretrained(p); tok.save_pretrained(p); saved[ep] = p
    ev = [round(h["eval_loss"],3) for h in tr.state.log_history if "eval_loss" in h]
    print(f"[{rc['name']}] eval_loss {ev} | {(time.time()-t0)/60:.1f} min | saved {sorted(saved)}")
    del tr, model, tok; gc.collect(); torch.cuda.empty_cache()
    return {"name":rc["name"], "eval_loss":ev, "saved":{str(k):str(v) for k,v in saved.items()}}""")

md(r"""## 6. (optional) One-arm smoke test

Sanity-check on the cheapest arm before committing hours to all four. Skip if confident.""")
code(r"""# _r = run_recipe(RECIPES[0]); _r   # uncomment to train ONLY the gentle arm first""")

md(r"""## 7. Train all 4 recipes — saves EVERY epoch to Drive

A failure in one arm is caught and the sweep continues; completed arms are already on Drive. **~2.5–3.5h H100.**
If your session is time-limited, comment out arms and run in two passes — the RUN tag keeps them side-by-side.""")
code(r"""RESULTS = []
for rc in RECIPES:
    print("="*72, f"\nTRAIN  {rc['name']}  {rc}")
    try:
        RESULTS.append(run_recipe(rc))
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"!! FAILED {rc['name']}: {e}")
print("\n"+"="*72+"\nTRAINING DONE — every epoch on Drive")
for r in RESULTS:
    print(f"  {r['name']:9s}  eval_loss {r['eval_loss']}  -> {sorted(r['saved'])}")
import json as _j; print("\nRESULTS =", _j.dumps(RESULTS, ensure_ascii=False))""")

md(r"""## 8. SCORE every checkpoint → write `speaker_scores_v3sweep.{json,md}` to Drive

Reloads each saved adapter (discovered from Drive — re-runnable even after a fresh runtime), runs a fixed
14-probe battery, and records BOTH the full answer AND automatic metrics:
- `prof` (profanity hits — his voice REQUIRES it), `stopped` (ended on <|im_end|> vs ran on = rambling),
- `distinct3` (unique-trigram ratio; low = degenerate/looping = collapse), `foreign` (CJK/Cyrillic/Arabic
  chars = hard collapse), `think_leak` (Qwen3 <think> escaped), `topic_hit` (stayed on the asked topic).
- `health` (0–100) = a rough COLLAPSE FILTER only. **Coherence + opinion-fidelity are judged by READING the
  outputs** (that's why every answer is in the file) — send the .md/.json back and I'll rank + pick the first to try.
Decoding is seeded (reproducible) sampling at the real inference settings, so profanity/voice show naturally.""")
code(r"""import re, json, glob, gc, torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

PROF    = re.compile(r"\b(sik|am[ıi]na|amc[ıi]k|g[öo]t|pi[çc]|o[çc]\b|orospu|yarr?a[kğ]|pezevenk|kahpe|ibne|gavat|anan[ıi]|avrad[ıi]|sokay[ıi]m|siktir|yavşak|şerefsiz)", re.I)
FOREIGN = re.compile(r"[一-鿿぀-ヿ가-힯Ѐ-ӿ؀-ۿ]")  # CJK/Kana/Hangul/Cyrillic/Arabic = collapse

# 14 probes: his BROKEN topics (opinion, known-covered) + voice holds + fact/confab baseline.
PROBES = [
    {"kind":"opinion","topic":"mutlak butlan","q":"Mutlak butlan meselesi hakkinda ne dusunuyorsun?","kw":["butlan"]},
    {"kind":"opinion","topic":"dersimli kk","q":"Kemal Kilicdaroglu'nun Dersimli olmasi meselesini nasil yorumluyorsun?","kw":["dersim","kilicdaroglu","kemal"]},
    {"kind":"opinion","topic":"almanci","q":"Almanci ne demek, gurbetciden farki ne?","kw":["almanci","gurbetci"]},
    {"kind":"opinion","topic":"12 eylul","q":"12 Eylul darbesi ve Kenan Evren hakkinda ne dusunuyorsun?","kw":["eylul","evren","darbe"]},
    {"kind":"opinion","topic":"ataturk","q":"Ataturk hakkinda ne dusunuyorsun?","kw":["ataturk","mustafa kemal"]},
    {"kind":"opinion","topic":"pkk sureci","q":"PKK acilim sureci hakkinda ne dusunuyorsun?","kw":["pkk","surec","acilim"]},
    {"kind":"opinion","topic":"ocalan","q":"Ocalan hakkinda ne dusunuyorsun?","kw":["ocalan","apo"]},
    {"kind":"opinion","topic":"hitler","q":"Hitler ve Nazizm hakkinda ne dusunuyorsun?","kw":["hitler","nazi"]},
    {"kind":"opinion","topic":"askeri vesayet","q":"Turkiye'de askeri vesayet meselesi neydi?","kw":["vesayet","ordu","asker"]},
    {"kind":"opinion","topic":"suriye","q":"Suriye politikamiz hakkinda ne dusunuyorsun?","kw":["suriye","esad"]},
    {"kind":"voice","topic":"chp","q":"CHP hakkinda ne dusunuyorsun?","kw":["chp"]},
    {"kind":"voice","topic":"milliyetcilik","q":"Turk milliyetciligi nedir sence?","kw":["milliyet","turk"]},
    {"kind":"fact","topic":"cumhuriyet yili","q":"Turkiye Cumhuriyeti kac yilinda kuruldu?","kw":["1923"]},
    {"kind":"fact","topic":"obscure fact","q":"Sabetay Sevi kac yilinda oldu?","kw":[]},
]

def distinct_n(text, n=3):
    t = text.split()
    if len(t) < n: return 1.0
    g = [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
    return len(set(g))/max(len(g),1)

def score_answer(a, stopped, raw, kw):
    d3 = distinct_n(a,3); prof = len(PROF.findall(a)); foreign = len(FOREIGN.findall(a))
    think = "<think>" in raw; hit = (any(k in a.lower() for k in kw) if kw else None)
    h = 100.0
    if think: h -= 40
    if foreign: h -= 30
    h -= 35*(1-d3)                 # degeneration / looping
    if not stopped: h -= 15
    if len(a) < 40: h -= 15
    return {"chars":len(a),"prof":prof,"has_prof":prof>0,"distinct3":round(d3,3),
            "foreign":foreign,"think_leak":think,"stopped":bool(stopped),"topic_hit":hit,
            "health":round(max(0.0,h),1)}

def gen(model, tok, q):
    torch.manual_seed(SEED)   # reproducible sampling -> comparable across models, profanity/voice still show
    msgs = [{"role":"system","content":SYS},{"role":"user","content":q}]
    inp = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=inp, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.85,
                         repetition_penalty=1.2, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    g = out[0][inp.shape[1]:]
    stopped = int(g[-1]) == tok.eos_token_id
    return tok.decode(g, skip_special_tokens=True).strip(), stopped, tok.decode(g, skip_special_tokens=False)

ckpts = sorted(glob.glob(f"{OUTPUT_ROOT}/qwen3_{RUN}_*_ep*"))
assert ckpts, f"no checkpoints under {OUTPUT_ROOT}/qwen3_{RUN}_*_ep* — run Section 7 first."
print(f"scoring {len(ckpts)} checkpoints x {len(PROBES)} probes ...")
report = {}
for adir in ckpts:
    name = os.path.basename(adir)
    m, tk = FastLanguageModel.from_pretrained(model_name=adir, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    tk = get_chat_template(tk, chat_template="qwen-2.5", map_eos_token=True)
    FastLanguageModel.for_inference(m)
    rows = []
    for p in PROBES:
        a, st, raw = gen(m, tk, p["q"])
        rows.append({"kind":p["kind"],"topic":p["topic"],"q":p["q"],"answer":a, **score_answer(a, st, raw, p["kw"])})
    n = len(rows); hits = [r for r in rows if r["topic_hit"] is not None]
    agg = {"health":round(sum(r["health"] for r in rows)/n,1),
           "prof_rate":round(sum(r["has_prof"] for r in rows)/n,2),
           "stop_rate":round(sum(r["stopped"] for r in rows)/n,2),
           "mean_distinct3":round(sum(r["distinct3"] for r in rows)/n,3),
           "think_leaks":sum(r["think_leak"] for r in rows),
           "foreign_answers":sum(r["foreign"]>0 for r in rows),
           "topic_hit_rate":(round(sum(r["topic_hit"] for r in hits)/len(hits),2) if hits else None),
           "mean_chars":int(sum(r["chars"] for r in rows)/n)}
    report[name] = {"agg":agg, "rows":rows}
    print(f"  {name:34s} health {agg['health']:5.1f} | prof {agg['prof_rate']} | stop {agg['stop_rate']} | d3 {agg['mean_distinct3']} | topic {agg['topic_hit_rate']}")
    del m, tk; gc.collect(); torch.cuda.empty_cache()

# --- write JSON (machine) + Markdown (readable, full outputs) to Drive ---
json.dump(report, open(f"{OUTPUT_ROOT}/speaker_scores_{RUN}.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
lb = sorted(report.items(), key=lambda kv:-kv[1]["agg"]["health"])
L = [f"# Speaker {RUN} — model scores ({len(report)} checkpoints)\n",
     "Auto `health` is a COLLAPSE FILTER only (penalizes <think> leak, foreign script, looping, no-stop).",
     "**Coherence + opinion-fidelity are judged by reading the outputs below** — send this file back to pick a winner.\n",
     "## Leaderboard (by auto health)\n",
     "| model | health | prof | stop | distinct3 | topic_hit | think | foreign | chars |",
     "|---|---|---|---|---|---|---|---|---|"]
for name,d in lb:
    a=d["agg"]; L.append(f"| {name} | {a['health']} | {a['prof_rate']} | {a['stop_rate']} | {a['mean_distinct3']} | {a['topic_hit_rate']} | {a['think_leaks']} | {a['foreign_answers']} | {a['mean_chars']} |")
L.append("\n---\n## Full outputs\n")
for name,d in report.items():
    L.append(f"### {name}  (health {d['agg']['health']}, prof {d['agg']['prof_rate']}, stop {d['agg']['stop_rate']})\n")
    for r in d["rows"]:
        L.append(f"**[{r['kind']}] {r['topic']}** — {r['q']}")
        L.append(f"> prof={r['prof']} stop={r['stopped']} d3={r['distinct3']} foreign={r['foreign']} think={r['think_leak']} health={r['health']}")
        L.append(f"\n{r['answer']}\n")
open(f"{OUTPUT_ROOT}/speaker_scores_{RUN}.md","w",encoding="utf-8").write("\n".join(L))
print(f"\nWROTE  {OUTPUT_ROOT}/speaker_scores_{RUN}.json  and  .md  -> download the .md and send it back.")""")

md(r"""## 9. What to send back

Download **`speaker_scores_v3sweep.md`** (readable, has every answer) from Drive and paste/send it here.
I'll rank the 10 checkpoints on his voice + coherence + opinion-fidelity (using `health` only to drop
collapsed ones) and tell you the single one to convert to GGUF and try first. Then run Section 10 for it.""")

md(r"""## 10. (run AFTER we pick) Export the winner(s) → Q4_K_M GGUF + Modelfile

Set `WINNERS` to the `(arm, epoch)` picks and run. Builds llama.cpp once; only Q4_K_M (what you run locally).""")
code(r"""WINNERS = []   # e.g. [("gentle", 2), ("capslow", 3)] — fill from the score-file pick, then run this cell
import glob, shutil, subprocess, gc
def _sh(c): print("$", c); subprocess.run(c, shell=True, check=True)
LCPP = "/content/llama.cpp"
if WINNERS and not os.path.isdir(LCPP):
    _sh(f"git clone -q https://github.com/ggml-org/llama.cpp {LCPP}")
    _sh("pip install -q gguf sentencepiece protobuf")
    _sh(f"cmake -S {LCPP} -B {LCPP}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /content/_lcpp.log 2>&1")
    _sh(f"cmake --build {LCPP}/build -j --target llama-quantize >> /content/_lcpp.log 2>&1")
def _qbin():
    for p in [f"{LCPP}/build/bin/llama-quantize", f"{LCPP}/build/llama-quantize"]:
        if os.path.exists(p): return p
    return glob.glob(f"{LCPP}/**/llama-quantize", recursive=True)[0]
TMPL = ("{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
        "{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n")

def export_arm(name, epoch):
    from unsloth import FastLanguageModel
    adir = f"{OUTPUT_ROOT}/qwen3_{RUN}_{name}_ep{epoch}"
    assert os.path.isdir(adir), f"missing {adir}"
    print("="*72, f"\nEXPORT {name} ep{epoch}")
    m2, t2 = FastLanguageModel.from_pretrained(model_name=adir, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    merged = f"/content/merged_{name}_ep{epoch}"
    m2.save_pretrained_merged(merged, t2, save_method="merged_16bit")
    del m2, t2; gc.collect(); torch.cuda.empty_cache()
    f16 = f"/content/{name}_ep{epoch}_f16.gguf"
    _sh(f'python {LCPP}/convert_hf_to_gguf.py "{merged}" --outfile "{f16}" --outtype f16')
    shutil.rmtree(merged, ignore_errors=True)
    gguf = f"speaker-qwen3-14b-{RUN}-{name}-ep{epoch}-q4_k_m.gguf"
    _sh(f'"{_qbin()}" "{f16}" "{OUTPUT_ROOT}/{gguf}" Q4_K_M'); os.remove(f16)
    mf = ["FROM ./" + gguf, 'TEMPLATE ' + '"'*3 + TMPL + '"'*3, 'SYSTEM ' + '"'*3 + SYS + '"'*3,
          "PARAMETER temperature 0.7", "PARAMETER top_p 0.85", "PARAMETER repeat_penalty 1.2",
          "PARAMETER num_ctx 4096", "PARAMETER num_predict 512",
          'PARAMETER stop "<|im_end|>"', 'PARAMETER stop "<|endoftext|>"', 'PARAMETER stop "<think>"']
    open(f"{OUTPUT_ROOT}/Modelfile-speaker-{name}-ep{epoch}","w",encoding="utf-8").write("\n".join(mf)+"\n")
    print("WROTE", f"{OUTPUT_ROOT}/{gguf}")

if not WINNERS:
    print("WINNERS is empty — set it from the score-file pick (e.g. [('gentle',2)]) and re-run this cell.")
for nm, ep in WINNERS:
    try: export_arm(nm, ep)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"!! export failed {nm} ep{ep}: {e}")""")

md(r"""## 11. Local run — create + test the winner

Download the winner's `.gguf` **and** its `Modelfile-speaker-<name>-ep<N>` into one folder, then (PowerShell):
```powershell
ollama create speaker-<name>-ep<N> -f Modelfile-speaker-<name>-ep<N>
ollama run speaker-<name>-ep<N> "Mutlak butlan hakkinda ne dusunuyorsun?"
```
Test **with RAG OFF** on the broken topics — mutlak butlan, Dersimli KK, Almancı, 12 Eylül, Atatürk, PKK süreci.
Judge: his crude voice + natural profanity, a COHERENT take (not word-salad), holds the topic, stops on its own.
`ui/speaker_studio.py` (port 7861) lists it in the LLM dropdown for chat + TTS read-back.
""")

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "H100"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
out = Path(__file__).with_name("qwen_voice_sft.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(CELLS)} cells)")
