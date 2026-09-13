# Locked worker build environments

`studio` and `voice` are dependency-only (`package = false`) uv projects used
only to build the separate PyInstaller workers in Task 10.  They intentionally
do not install this repository as an editable package: the later PyInstaller
build supplies the checked repository code explicitly.

The Studio lock has the CPU-only `torch==2.7.1+cpu` from the explicit PyTorch
CPU index.  It owns Gradio and retrieval (`qdrant-client`, FlagEmbedding,
rank-bm25), and deliberately excludes XTTS, FAISS, torchaudio, and the Applio
stack.  The voice lock has `torch==2.7.1+cu128` and
`torchaudio==2.7.1+cu128` exclusively from the explicit CUDA 12.8 index.  It
owns XTTS, RVC/Applio inference dependencies, and FAISS, and deliberately
excludes Gradio and the retrieval stack.

Both projects require CPython 3.11.  Recreate them from their committed locks:

```powershell
$env:PYTHONNOUSERSITE = "1"
$env:UV_PROJECT_ENVIRONMENT = "$PWD\.venv-build-studio"
uv sync --project packaging/runtime/studio --frozen
$env:UV_PROJECT_ENVIRONMENT = "$PWD\.venv-build-voice"
uv sync --project packaging/runtime/voice --frozen
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

Use `python -I` for worker import smoke tests.  Isolated mode deliberately
omits the current directory, so the checked repository root must be inserted
explicitly before importing the local worker module; this is a smoke-test
mechanism, not an editable install:

```powershell
$repo = (Get-Location).Path
& .venv-build-studio\Scripts\python.exe -I -c "import sys; sys.path.insert(0, r'$repo'); import gradio, qdrant_client, FlagEmbedding, ui.server"
& .venv-build-voice\Scripts\python.exe -I -c "import sys; sys.path.insert(0, r'$repo'); import torch, TTS, faiss, voice.server"
```

The voice lock uses `coqui-tts==0.26.2`, the distribution that provides the
`TTS` package.  Its upstream metadata requires `transformers>=4.47,<4.52`, so
the lock deliberately uses `transformers==4.48.3` rather than the incompatible
`5.4.0` observed in the previous ad-hoc RVC environment.  The Applio source is
not a PyPI dependency: Task 10 stages the audited inference-only source pinned
by `third_party/applio.lock.json` at commit
`3e5e248c2bd003f65f3e128dd12677533def27be`.  ContentVec and RMVPE weights are
voice-artifact inputs and are never downloaded while resolving these locks.
