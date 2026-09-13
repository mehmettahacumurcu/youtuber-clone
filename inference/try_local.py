"""Run the Speaker pilot LoRA locally (no Colab) via peft + bitsandbytes 4-bit.

This is the SIMPLEST no-Colab path: no GGUF, no merge, no llama.cpp. It loads the exact same
4-bit base the adapter was trained on + your adapter, and generates. Fits an 8 GB GPU (RTX 3060 Ti);
bitsandbytes ships Windows CUDA wheels and supports Ampere (sm_86).

One-time setup (PowerShell):
    py -3.11 -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install -U transformers accelerate peft bitsandbytes

Usage (point --adapter at the UNZIPPED speaker_lora folder):
    python inference/try_local.py --adapter "E:\\path\\to\\speaker_lora" --prompt "Abi ya, "

It's a BASE/continuation model: seed the START of a line in his register, not an instruction.
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Pre-quantized 4-bit base (ungated, ~5.5 GB download) — the exact model the adapter was trained on.
BASE = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate with the Speaker LoRA adapter, locally")
    ap.add_argument("--adapter", required=True, help="path to the unzipped speaker_lora folder")
    ap.add_argument("--prompt", default="Abi ya, ")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter)  # tokenizer was saved alongside the adapter
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")  # bitsandbytes reads the 4-bit config
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
