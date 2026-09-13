"""Generator for training/export_cosmos_gguf.ipynb.

Produces a self-contained Colab notebook: Run-all loads a saved cosmos LoRA
adapter from Drive, merges it into the base, converts to a Q4_K_M GGUF, and
drops the .gguf + a matching Ollama Modelfile back on Drive for download.

Run:  python training/build_export_nb.py   ->   training/export_cosmos_gguf.ipynb
"""
import json
from pathlib import Path

CELLS = []

def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src})

# ---------------------------------------------------------------- 1. title
md(r"""# Export cosmos adapter -> GGUF for local Ollama

**Run-all and you get a model.** This loads a saved cosmos LoRA checkpoint from
Drive, merges it into the base, quantizes to **Q4_K_M** GGUF (fits an 8 GB GPU),
and writes the `.gguf` + a matching **Modelfile** to
`MyDrive/speaker_models/`. Download those two files and `ollama create` locally.

- Use a **GPU runtime** (Runtime -> Change runtime type -> T4 is enough).
- Default exports **all three doses (epochs 3, 4, 5)** -> three GGUFs + Modelfiles.
  Trim `EPOCHS` in the config cell to e.g. `[3]` to do just one.
- If a merge runs out of RAM on a free T4, switch to **High-RAM** runtime, or do
  one epoch at a time.
""")

# ---------------------------------------------------------------- 2. install
code(r"""%%capture
!pip install -q unsloth
# only used by the fallback path (manual llama.cpp convert); cheap to have ready
!pip install -q gguf protobuf sentencepiece""")

# ---------------------------------------------------------------- 3. mount + config
code(r"""from google.colab import drive
drive.mount('/content/drive')

import os
MODEL_DIR   = "/content/drive/MyDrive/speaker_models"
NAME        = "cosmos-turkish-8b"
EPOCHS      = [3, 4, 5]    # which checkpoints to export; trim to e.g. [3] to do one
QUANT       = "q4_k_m"     # 8 GB-friendly; "q5_k_m" for a bit more quality if you have room
MAX_SEQ_LEN = 2048

for e in EPOCHS:
    p = f"{MODEL_DIR}/{NAME}_lora_ep{e}"
    assert os.path.exists(p), f"adapter not found on Drive: {p}"
print("will export:", [f"{NAME}_lora_ep{e}" for e in EPOCHS], "->", QUANT)""")

# ---------------------------------------------------------------- 4. helpers + export fn
# NOTE: no literal triple-quote appears in this cell (the Modelfile TEMPLATE line is
# assembled with '"'*3) so the generator's own r\"\"\"...\"\"\" wrapping stays intact.
code(r"""import gc, glob, shutil, subprocess, torch
from unsloth import FastLanguageModel

def free():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

def sh(cmd):
    print("$", cmd); subprocess.run(cmd, shell=True, check=True)

def write_modelfile(epoch, gguf_name):
    # raw passthrough template -> feeds the prompt verbatim (this is a BASE/transcript
    # completion model; the default Llama-3 chat template would mismatch training).
    lines = [
        "FROM ./" + gguf_name,
        "TEMPLATE " + '"'*3 + "{{ .Prompt }}" + '"'*3,
        "PARAMETER temperature 0.8",
        "PARAMETER top_p 0.9",
        "PARAMETER repeat_penalty 1.1",
        "PARAMETER num_predict 256",
        'PARAMETER stop "<|end_of_text|>"',
        'PARAMETER stop "<|eot_id|>"',
    ]
    path = f"{MODEL_DIR}/Modelfile-{NAME}-ep{epoch}"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path

def export_one(epoch):
    adapter = f"{MODEL_DIR}/{NAME}_lora_ep{epoch}"
    out_gguf = f"{MODEL_DIR}/{NAME}-ep{epoch}-{QUANT}.gguf"
    gguf_name = os.path.basename(out_gguf)
    print("\n" + "="*70 + f"\nEXPORT ep{epoch}\n  adapter: {adapter}\n  -> {out_gguf}\n" + "="*70)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=adapter, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)

    try:
        # primary: Unsloth one-shot merge + convert + quantize
        tmp = f"/content/gguf_ep{epoch}"
        model.save_pretrained_gguf(tmp, tok, quantization_method=QUANT)
        hits = glob.glob(f"{tmp}/*{QUANT.upper()}*.gguf") or glob.glob(f"{tmp}/*.gguf")
        assert hits, "no .gguf produced by save_pretrained_gguf"
        shutil.copy(hits[0], out_gguf)
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as ex:
        print("\n[primary GGUF path failed -> manual llama.cpp fallback]\n", repr(ex))
        merged = f"/content/merged_ep{epoch}"
        model.save_pretrained_merged(merged, tok, save_method="merged_16bit")
        if not os.path.exists("/content/llama.cpp"):
            sh("git clone -q https://github.com/ggerganov/llama.cpp /content/llama.cpp")
            sh("cmake -S /content/llama.cpp -B /content/llama.cpp/build -DGGML_CUDA=OFF "
               "-DLLAMA_CURL=OFF > /content/_cmake.log 2>&1")
            sh("cmake --build /content/llama.cpp/build -j --target llama-quantize "
               ">> /content/_cmake.log 2>&1")
        f16 = f"/content/f16_ep{epoch}.gguf"
        sh(f"python /content/llama.cpp/convert_hf_to_gguf.py {merged} --outfile {f16} --outtype f16")
        sh(f'/content/llama.cpp/build/bin/llama-quantize "{f16}" "{out_gguf}" Q4_K_M')
        os.remove(f16); shutil.rmtree(merged, ignore_errors=True)

    mf = write_modelfile(epoch, gguf_name)
    del model, tok; free()
    size_gb = round(os.path.getsize(out_gguf) / 1e9, 2)
    print(f"\nDONE ep{epoch}: {gguf_name} ({size_gb} GB) + {os.path.basename(mf)}")
    return out_gguf, mf""")

# ---------------------------------------------------------------- 5. run
code(r"""produced = []
for e in EPOCHS:
    produced.append(export_one(e))
print("\nALL EXPORTS DONE:")
for g, m in produced:
    print("  ", os.path.basename(g), "+", os.path.basename(m))""")

# ---------------------------------------------------------------- 6. verify + next steps
code(r"""print("Files now on your Drive (download these):\n")
sh(f'ls -lh "{MODEL_DIR}"/*.gguf "{MODEL_DIR}"/Modelfile-* 2>/dev/null')
print("\nNext, locally:")
for e in EPOCHS:
    print(f"  ollama create cosmos-speaker-ep{e} -f Modelfile-{NAME}-ep{e}")
    print(f"  ollama run cosmos-speaker-ep{e}")""")

# ---------------------------------------------------------------- 7. local steps (md)
md(r"""## Run it locally (Ollama)

1. From `MyDrive/speaker_models/` download the **`.gguf`** and its matching
   **`Modelfile-cosmos-turkish-8b-ep3`** into the *same* local folder
   (e.g. `E:\youtuber-clone\models\`).
2. In PowerShell, from that folder:
   ```powershell
   ollama create cosmos-speaker-ep3 -f Modelfile-cosmos-turkish-8b-ep3
   ollama run cosmos-speaker-ep3
   ```
3. At the `>>>` prompt, give it a transcript-style opener and let him continue:
   ```
   Değerli dostlar, bugün size çok önemli bir meseleden bahsedeceğim.
   ```

For a precise base-completion test (no chat quirks), use the raw API:
```powershell
curl http://localhost:11434/api/generate -d '{\"model\":\"cosmos-speaker-ep3\",\"prompt\":\"Değerli dostlar, bugün size\",\"raw\":true,\"stream\":false}'
```

Listen for: his tics (yani/abi/bak/işte), the livestream register, profanity
landing naturally, his topics, and whether it **stops on its own**.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

out = Path(__file__).with_name("export_cosmos_gguf.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"wrote {out}  ({len(CELLS)} cells)")
