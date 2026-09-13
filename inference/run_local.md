# Run the Speaker pilot model locally (no Colab)

You have `speaker_lora.zip` (a **LoRA adapter** — a ~150 MB patch, *not* a standalone model) and an
RTX 3060 Ti (8 GB). Using it always = **base Llama-3.1-8B + your adapter**. Two local paths.

**First, unzip it** and note the absolute path of the resulting `speaker_lora` folder (it has
`adapter_config.json` + `adapter_model.safetensors`). Below it's written as `E:\path\to\speaker_lora`.

---

## Path A — Ollama (you already have it)

Ollama needs a **GGUF adapter**; yours is safetensors, so convert it once with llama.cpp.

```powershell
# 1) llama.cpp + its converter deps (CPU-only; the 8 GB GPU is irrelevant for conversion)
git clone https://github.com/ggml-org/llama.cpp C:\llama.cpp
cd C:\llama.cpp
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements\requirements-convert_lora_to_gguf.txt

# 2) Convert LoRA -> GGUF. CRITICAL: override the base id. The adapter records
#    "unsloth/Meta-Llama-3.1-8B-bnb-4bit" (4-bit) as its base; feeding that to the converter
#    produces GARBAGE. Force the ungated 16-bit config instead (downloads config only, no weights):
python convert_lora_to_gguf.py `
  --base-model-id unsloth/Meta-Llama-3.1-8B `
  --outtype f16 `
  --outfile C:\llama.cpp\speaker-lora-f16.gguf `
  "E:\path\to\speaker_lora"

# 3) Pull the BASE (text/completion) model — NOT llama3.1:8b, that's Instruct and will misbehave.
ollama pull llama3.1:8b-text-q4_K_M
```

Create a file named `Modelfile` (no extension):

```dockerfile
FROM llama3.1:8b-text-q4_K_M
ADAPTER C:\llama.cpp\speaker-lora-f16.gguf

# Base/completion model: NO chat template — pass the prompt through raw, or Ollama's default
# Instruct scaffolding will mangle generation.
TEMPLATE """{{ .Prompt }}"""

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<|end_of_text|>"
```

```powershell
ollama create speaker -f Modelfile
ollama run speaker "Abi ya, "
```

VRAM: q4_K_M base (~4.9 GB) + tiny adapter fits 8 GB comfortably.

---

## Path B — Python (fewer steps, most faithful to training)

No GGUF, no merge, no llama.cpp. Loads the same 4-bit base + adapter you trained.
Script: [`inference/try_local.py`](try_local.py).

```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -U transformers accelerate peft bitsandbytes
python inference/try_local.py --adapter "E:\path\to\speaker_lora" --prompt "Abi ya, "
```

(bitsandbytes has official Windows CUDA wheels; the 3060 Ti / Ampere is supported. First run
downloads the ~5.5 GB 4-bit base once.)

---

## Gotchas (both paths)
- **Base, not Instruct.** Use `llama3.1:8b-text-*` (Path A) / the bnb-4bit base (Path B). An Instruct base under a base-trained adapter is the real cause of "erratic" output.
- **bnb-4bit trap (Path A only):** never let the converter auto-read the adapter's base — pass `--base-model-id unsloth/Meta-Llama-3.1-8B`.
- **No chat template:** it's a continuation model. Seed the **start of a line in his register**, e.g.
  `"Oğlum bak, bu dizinin olayı şu: "` — not an instruction like "explain X".
- **Expectations:** this is the pilot adapter (498k tokens, light touch) — voice is present but mild and a
  bit formal. Stronger voice comes after the contamination cleanup + full-corpus run.
