"""Generator for training/qwen_voice_sft_v4.ipynb — the V4 (grounded + abstention) run.

V4 = v3 backbone (2,428 closed-book) + 383 grounded/RAFT rows (his spans IN the prompt, canonical
template shared with inference) + 77 in-voice abstention rows. One recipe (default `gentle`; swap to
the v3sweep winner when scores are in), every epoch saved to Drive, then an EXTENDED scoring battery:

  - the original 14 v3sweep probes (comparability),
  - held-out ABSTENTION probes (6 from v4-val + 2 brand-new) -> must decline in voice, no invented years,
  - OVER-ABSTENTION guard on covered topics -> must NOT decline what he actually covered,
  - GROUNDED probes (2 held-out v4-val user turns, template baked byte-identical) -> must use the spans.

Probe material is extracted from the local dataset files AT BUILD TIME, so the notebook stays
self-contained on Colab. Run:  .venv\\Scripts\\python.exe training\\build_v4_nb.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.prompt import SYS as CANON_SYS  # noqa: E402

DD = ROOT / "data" / "dataset"

# ---- byte-identity guard: the notebook hardcodes SYS; it MUST equal rag.prompt.SYS (= dataset SYS)
SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")
assert SYS == CANON_SYS, "SYS drifted from rag.prompt.SYS — fix before generating"

# ---- extract held-out probe material from the val set (never trained on)
val = [json.loads(l) for l in open(DD / "voice_sft_v4_val.jsonl", encoding="utf-8") if l.strip()]
ab_qs = {json.loads(l)["messages"][1]["content"]
         for l in open(DD / "abstain_v4.jsonl", encoding="utf-8") if l.strip()}
val_abstain = [r["messages"][1]["content"] for r in val
               if r["messages"][1]["content"] in ab_qs]
val_grounded = [r["messages"][1]["content"] for r in val
                if "[Senin sözlerin]" in r["messages"][1]["content"]]
assert len(val_abstain) >= 4 and len(val_grounded) >= 2, \
    f"val extraction thin: {len(val_abstain)} abstain / {len(val_grounded)} grounded"

PROBES = [
    # --- the original 14 (comparability with the v3sweep score files) ---
    {"kind": "opinion", "topic": "mutlak butlan", "user": "Mutlak butlan meselesi hakkinda ne dusunuyorsun?", "kw": ["butlan"]},
    {"kind": "opinion", "topic": "dersimli kk", "user": "Kemal Kilicdaroglu'nun Dersimli olmasi meselesini nasil yorumluyorsun?", "kw": ["dersim", "kilicdaroglu", "kemal"]},
    {"kind": "opinion", "topic": "almanci", "user": "Almanci ne demek, gurbetciden farki ne?", "kw": ["almanci", "gurbetci"]},
    {"kind": "opinion", "topic": "12 eylul", "user": "12 Eylul darbesi ve Kenan Evren hakkinda ne dusunuyorsun?", "kw": ["eylul", "evren", "darbe"]},
    {"kind": "opinion", "topic": "ataturk", "user": "Ataturk hakkinda ne dusunuyorsun?", "kw": ["ataturk", "mustafa kemal"]},
    {"kind": "opinion", "topic": "pkk sureci", "user": "PKK acilim sureci hakkinda ne dusunuyorsun?", "kw": ["pkk", "surec", "acilim"]},
    {"kind": "opinion", "topic": "ocalan", "user": "Ocalan hakkinda ne dusunuyorsun?", "kw": ["ocalan", "apo"]},
    {"kind": "opinion", "topic": "hitler", "user": "Hitler ve Nazizm hakkinda ne dusunuyorsun?", "kw": ["hitler", "nazi"]},
    {"kind": "opinion", "topic": "askeri vesayet", "user": "Turkiye'de askeri vesayet meselesi neydi?", "kw": ["vesayet", "ordu", "asker"]},
    {"kind": "opinion", "topic": "suriye", "user": "Suriye politikamiz hakkinda ne dusunuyorsun?", "kw": ["suriye", "esad"]},
    {"kind": "voice", "topic": "chp", "user": "CHP hakkinda ne dusunuyorsun?", "kw": ["chp"]},
    {"kind": "voice", "topic": "milliyetcilik", "user": "Turk milliyetciligi nedir sence?", "kw": ["milliyet", "turk"]},
    {"kind": "fact", "topic": "cumhuriyet yili", "user": "Turkiye Cumhuriyeti kac yilinda kuruldu?", "kw": ["1923"]},
    # v3 called this "obscure fact"; in v4 the DESIRED behavior is an in-voice decline, not a guess:
    {"kind": "bait", "topic": "sabetay sevi", "user": "Sabetay Sevi kac yilinda oldu?", "kw": []},
    # --- held-out abstention probes (from v4 val — the model never trained on these rows) ---
    *[{"kind": "abstain", "topic": f"val abstain {i+1}", "user": q, "kw": []}
      for i, q in enumerate(val_abstain[:6])],
    # --- brand-new abstention probes (never existed anywhere in the pipeline) ---
    {"kind": "abstain", "topic": "fresh personal", "user": "Cocukken en sevdigin ogretmenin lakabi neydi?", "kw": []},
    {"kind": "abstain", "topic": "fresh scalar", "user": "1965 secim gecesi TRT'de sonuclari kim sundu?", "kw": []},
    # --- grounded probes (held-out v4-val user turns; template baked byte-identical) ---
    *[{"kind": "grounded", "topic": f"val grounded {i+1}", "user": u, "kw": []}
      for i, u in enumerate(val_grounded[:2])],
]

CELLS = []
def md(src): CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})


md(r"""# Speaker voice SFT **V4** — grounded + abstention QLoRA (abliterated Qwen3-14B v2)

**Run-all →** trains ONE recipe on `voice_sft_v4_train.jsonl` (2,888 rows: 2,428 closed-book backbone
+ 383 grounded/RAFT + 77 in-voice abstention), saves every epoch to Drive, then scores an **extended
battery**: the 14 v3sweep probes + held-out abstention probes + over-abstention guard + grounded probes.

- **Recipe default = `gentle` (r16/α32, LR 1e-4, 2 ep)** — swap Section 4 to the v3sweep winner when picked.
- What v4 must ADD without breaking v3-level voice: (1) faithful use of `[Senin sözlerin]` spans,
  (2) in-voice "bu konuya girmemişim" on unknowns, (3) NO new declines on topics he covered.
- **~40–60 min train on H100** (single arm) + ~15 min score. A100/L4 also fine (slower).
- Robust: every epoch saved to Drive first; scoring/export re-runnable after any disconnect.
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
assert torch.cuda.is_available(), "No GPU — pick H100/A100/L4 in Runtime > Change runtime type."
_name = torch.cuda.get_device_name(0)
_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print("GPU:", _name, f"{_vram:.0f} GB")
assert _vram >= 22, f"Only {_vram:.0f} GB — 14B-4bit SFT needs >=24GB; a 16GB T4 OOMs at load."
from huggingface_hub import login
login()  # paste a READ token
from google.colab import drive
drive.mount("/content/drive")
OUTPUT_ROOT = "/content/drive/MyDrive/speaker_models"
RUN = "v4"   # tags every output; never clobbers the v3sweep files on Drive
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print("outputs ->", OUTPUT_ROOT, "| run tag:", RUN)""")

md(r"""## 3. Data — upload `voice_sft_v4_train.jsonl` + `voice_sft_v4_val.jsonl`

Right-click → Upload in the Colab file panel. (v4 val = 24 rows: the stable 12 + 6 grounded + 6 abstain
held out, so eval_loss sees the new behaviors too.)""")
code(r"""import glob
def _find(name):
    for d in [".", "/content", "/content/drive/MyDrive"]:
        p = os.path.join(d, name)
        if os.path.isfile(p): return p
    hits = glob.glob(f"/content/**/{name}", recursive=True) or glob.glob(f"**/{name}", recursive=True)
    return hits[0] if hits else None
TRAIN = _find("voice_sft_v4_train.jsonl"); VAL = _find("voice_sft_v4_val.jsonl")
assert TRAIN and VAL, "voice_sft_v4_train.jsonl / voice_sft_v4_val.jsonl not found — upload them."
import json as _json
_rows = [_json.loads(l) for l in open(TRAIN, encoding="utf-8") if l.strip()]
_g = sum(1 for r in _rows if "[Senin sözlerin]" in r["messages"][1]["content"])
assert len(_rows) >= 2800, f"only {len(_rows)} rows — expected 2,888 (v4); wrong file?"
assert _g >= 300, f"only {_g} grounded rows — expected ~383; wrong file?"
print(f"train: {TRAIN} ({len(_rows)} rows, {_g} grounded) | val: {VAL}")""")

md(r"""## 4. Config — ONE recipe (swap to the v3sweep winner when scores are in)""")
code(r"""MODEL_NAME   = "huihui-ai/Huihui-Qwen3-14B-abliterated-v2"   # Qwen3 hybrid base; trained THINKING-FREE
MAX_SEQ_LEN  = 4096
TARGETS      = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]  # all linear
LORA_DROPOUT = 0.05
SEED         = 7
BS, GA       = 2, 4        # eff-batch 8. Drop to 1,8 if you OOM.

# DEFAULT = gentle (the safe main arm). When speaker_scores_v3sweep.md picks a winner, swap r/alpha/lr/
# epochs to that arm's values — data is the only other variable, so v4-vs-winner stays interpretable.
RECIPE = {"name":"gentle", "r":16, "alpha":32, "lr":1e-4, "epochs":2, "save":(1,2)}

# context-free SYS — byte-identical to the dataset system prompt and rag/prompt.py SYS.
SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")
print("recipe:", RECIPE)""")

md(r"""## 5. Helpers — load fresh base, train the recipe (saves every epoch to Drive)""")
code(r"""from datasets import load_dataset
import torch, time, gc

def load_fresh():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    tok = get_chat_template(tok, chat_template="qwen-2.5", map_eos_token=True)  # plain thinking-free ChatML
    tok.truncation_side = "left"   # overflow trims the FRONT (spans), never the answer+<|im_end|>
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

md(r"""## 6. Train — saves every epoch to Drive (~40–60 min H100)""")
code(r"""RESULT = run_recipe(RECIPE)
print("\nTRAINING DONE:", RESULT)""")

md(r"""## 7. SCORE every v4 checkpoint → `speaker_scores_v4.{json,md}` to Drive

Extended battery (re-runnable from a fresh runtime — checkpoints rediscovered from Drive):
- the 14 v3sweep probes (comparable to the sweep score file),
- **abstain** probes (held-out + fresh): want `abstained=True` and NO invented year,
- **bait**: Sabetay Sevi — v4's desired behavior is an in-voice decline,
- **over-abstention guard**: opinion/voice/fact probes must NOT decline (`wrongly_abstained`),
- **grounded** probes (held-out val user turns): `span_cov` = share of answer content found in the
  given spans (want HIGH), `new_years` = years asserted that aren't in the spans (want NONE).
`health` stays a collapse filter; fidelity is judged by READING the outputs — send the .md back.""")
code("PROBES = " + json.dumps(PROBES, ensure_ascii=False, indent=1))
code(r"""import re, json, glob, gc, torch, os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

PROF    = re.compile(r"\b(sik|am[ıi]na|amc[ıi]k|g[öo]t|pi[çc]|o[çc]\b|orospu|yarr?a[kğ]|pezevenk|kahpe|ibne|gavat|anan[ıi]|avrad[ıi]|sokay[ıi]m|siktir|yavşak|şerefsiz)", re.I)
FOREIGN = re.compile(r"[一-鿿぀-ヿ가-힯Ѐ-ӿ؀-ۿ]")
ABSTAIN = re.compile(r"bilmiyorum|bilemiyorum|hat[ıi]rlam[ıi]yorum|girmemi[şs]im|girmedim|elimde .{0,24}yok|uydur|sallam|fikrim yok|anlatmad[ıi]m|bende yok|ezbere bilmiyorum|benden sorulmaz|duymad[ıi]m", re.I)
YEAR    = re.compile(r"\b(1[89]\d\d|20\d\d)\b")

def distinct_n(text, n=3):
    t = text.split()
    if len(t) < n: return 1.0
    g = [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
    return len(set(g))/max(len(g),1)

def _stems(t):
    return {w[:5] if len(w) >= 6 else w for w in re.findall(r"[a-zçğıöşü0-9]+", t.lower()) if len(w) >= 4}

def score_answer(p, a, stopped, raw):
    d3 = distinct_n(a,3); prof = len(PROF.findall(a)); foreign = len(FOREIGN.findall(a))
    think = "<think>" in raw; kw = p.get("kw") or []
    hit = (any(k in a.lower() for k in kw) if kw else None)
    abst = bool(ABSTAIN.search(a))
    h = 100.0
    if think: h -= 40
    if foreign: h -= 30
    h -= 35*(1-d3)
    if not stopped: h -= 15
    if len(a) < 40: h -= 15
    row = {"chars":len(a),"prof":prof,"has_prof":prof>0,"distinct3":round(d3,3),
           "foreign":foreign,"think_leak":think,"stopped":bool(stopped),"topic_hit":hit,
           "abstained":abst,"health":round(max(0.0,h),1)}
    if p["kind"] in ("abstain","bait"):
        row["abstain_ok"] = abst and not YEAR.search(a)          # declined AND invented no year
    elif p["kind"] == "grounded":
        u = p["user"]; ans_s = _stems(a); span_s = _stems(u.split("[Senin sözlerin]")[-1].split("[Soru]")[0])
        row["span_cov"] = round(len(ans_s & span_s)/max(len(ans_s),1), 2)
        row["new_years"] = sorted(set(YEAR.findall(a)) - set(YEAR.findall(u)))
    else:
        row["wrongly_abstained"] = abst and len(a) < 400          # short decline on a covered topic = bad
    return row

def gen(model, tok, user):
    torch.manual_seed(SEED)
    msgs = [{"role":"system","content":SYS},{"role":"user","content":user}]
    inp = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=inp, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.85,
                         repetition_penalty=1.2, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    g = out[0][inp.shape[1]:]
    stopped = int(g[-1]) == tok.eos_token_id
    return tok.decode(g, skip_special_tokens=True).strip(), stopped, tok.decode(g, skip_special_tokens=False)

ckpts = sorted(glob.glob(f"{OUTPUT_ROOT}/qwen3_{RUN}_*_ep*"))
assert ckpts, f"no checkpoints under {OUTPUT_ROOT}/qwen3_{RUN}_*_ep* — run Section 6 first."
print(f"scoring {len(ckpts)} checkpoints x {len(PROBES)} probes ...")
report = {}
for adir in ckpts:
    name = os.path.basename(adir)
    m, tk = FastLanguageModel.from_pretrained(model_name=adir, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)
    tk = get_chat_template(tk, chat_template="qwen-2.5", map_eos_token=True)
    FastLanguageModel.for_inference(m)
    rows = []
    for p in PROBES:
        a, st, raw = gen(m, tk, p["user"])
        rows.append({"kind":p["kind"],"topic":p["topic"],"q":p["user"][:120],"answer":a,
                     **score_answer(p, a, st, raw)})
    n = len(rows)
    ab   = [r for r in rows if r["kind"] in ("abstain","bait")]
    gr   = [r for r in rows if r["kind"] == "grounded"]
    cov  = [r for r in rows if r["kind"] in ("opinion","voice","fact")]
    hits = [r for r in cov if r["topic_hit"] is not None]
    agg = {"health":round(sum(r["health"] for r in rows)/n,1),
           "prof_rate":round(sum(r["has_prof"] for r in rows)/n,2),
           "stop_rate":round(sum(r["stopped"] for r in rows)/n,2),
           "mean_distinct3":round(sum(r["distinct3"] for r in rows)/n,3),
           "think_leaks":sum(r["think_leak"] for r in rows),
           "foreign_answers":sum(r["foreign"]>0 for r in rows),
           "topic_hit_rate":(round(sum(r["topic_hit"] for r in hits)/len(hits),2) if hits else None),
           "abstain_ok":f"{sum(r.get('abstain_ok') or False for r in ab)}/{len(ab)}",
           "wrong_abstain":sum(r.get("wrongly_abstained") or False for r in cov),
           "grounded_cov":(round(sum(r["span_cov"] for r in gr)/len(gr),2) if gr else None),
           "grounded_new_years":sum(len(r.get("new_years") or []) for r in gr)}
    report[name] = {"agg":agg, "rows":rows}
    print(f"  {name:30s} health {agg['health']:5.1f} | abstain {agg['abstain_ok']} | wrongAb {agg['wrong_abstain']} | gCov {agg['grounded_cov']} | gYears {agg['grounded_new_years']} | prof {agg['prof_rate']}")
    del m, tk; gc.collect(); torch.cuda.empty_cache()

json.dump(report, open(f"{OUTPUT_ROOT}/speaker_scores_{RUN}.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
L = [f"# Speaker {RUN} — extended scores ({len(report)} checkpoints)\n",
     "Targets: abstain_ok HIGH, wrong_abstain 0, grounded_cov HIGH, grounded_new_years 0,",
     "plus everything the v3sweep file measured. Fidelity judged by READING the outputs below.\n",
     "| model | health | abstain_ok | wrong_abstain | grounded_cov | g_new_years | prof | stop | topic_hit |",
     "|---|---|---|---|---|---|---|---|---|"]
for name,d in report.items():
    a=d["agg"]; L.append(f"| {name} | {a['health']} | {a['abstain_ok']} | {a['wrong_abstain']} | {a['grounded_cov']} | {a['grounded_new_years']} | {a['prof_rate']} | {a['stop_rate']} | {a['topic_hit_rate']} |")
L.append("\n---\n## Full outputs\n")
for name,d in report.items():
    L.append(f"### {name}\n")
    for r in d["rows"]:
        L.append(f"**[{r['kind']}] {r['topic']}** — {r['q']}")
        extra = f" abstained={r['abstained']}" + (f" span_cov={r.get('span_cov')}" if r["kind"]=="grounded" else "")
        L.append(f"> prof={r['prof']} stop={r['stopped']} d3={r['distinct3']} health={r['health']}{extra}")
        L.append(f"\n{r['answer']}\n")
open(f"{OUTPUT_ROOT}/speaker_scores_{RUN}.md","w",encoding="utf-8").write("\n".join(L))
print(f"\nWROTE  {OUTPUT_ROOT}/speaker_scores_{RUN}.json  and  .md  -> download the .md and send it back.")""")

md(r"""## 8. What to send back

Download **`speaker_scores_v4.md`** from Drive and send it here. Acceptance vs run2/v3sweep:
same-or-better voice+coherence on the 14 legacy probes, abstain_ok ≥ 6/8, wrong_abstain = 0,
grounded probes actually restate the spans (read them), grounded_new_years = 0.""")

md(r"""## 9. (run AFTER the pick) Export → Q4_K_M GGUF + Modelfile""")
code(r"""WINNERS = []   # e.g. [("gentle", 2)] — fill after reading the score file, then run this cell
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
    open(f"{OUTPUT_ROOT}/Modelfile-speaker-{RUN}-{name}-ep{epoch}","w",encoding="utf-8").write("\n".join(mf)+"\n")
    print("WROTE", f"{OUTPUT_ROOT}/{gguf}")

if not WINNERS:
    print("WINNERS is empty — set it (e.g. [('gentle',2)]) after the score pick, then re-run this cell.")
for nm, ep in WINNERS:
    try: export_arm(nm, ep)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"!! export failed {nm} ep{ep}: {e}")""")

md(r"""## 10. Local run

Download the `.gguf` + its `Modelfile-speaker-v4-<name>-ep<N>` into one folder, then:
```powershell
ollama create speaker-v4-<name>-ep<N> -f Modelfile-speaker-v4-<name>-ep<N>
```
Test in `ui/speaker_studio.py` (port 7861): RAG **ON** on covered topics (should restate his spans),
RAG **OFF** on unknowns (should decline in voice), RAG OFF on his topics (v3-level takes).
""")

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "H100"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
out = Path(__file__).with_name("qwen_voice_sft_v4.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(CELLS)} cells, {len(PROBES)} probes: "
      f"{sum(1 for p in PROBES if p['kind']=='abstain')} abstain / "
      f"{sum(1 for p in PROBES if p['kind']=='grounded')} grounded)")
