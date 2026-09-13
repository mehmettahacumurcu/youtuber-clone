# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


repo = Path(SPECPATH).resolve().parents[1]
hooks = repo / "packaging" / "pyinstaller" / "hooks"
vendor = repo / "build" / "vendor" / "applio"
if not vendor.is_dir():
    raise SystemExit("verified Applio staging tree is required")
def runtime_data(item):
    source = Path(item[0])
    parts = {part.casefold() for part in source.parts}
    return not parts.intersection({"demos", "notebooks", "test", "tests", "testing", "test_data"}) and source.suffix.casefold() not in {
        ".ipynb", ".ckpt", ".gguf", ".index", ".pth", ".pt", ".safetensors",
    }


datas = [item for item in collect_data_files("TTS") if runtime_data(item)] + [(str(vendor), "vendor/applio")]
hiddenimports = [
    "TTS.tts.configs.xtts_config",
    "TTS.tts.models.xtts",
    "faiss",
    "faiss.swigfaiss_avx2",
    "noisereduce",
    "pedalboard",
    "torchcrepe",
    "torchfcpe",
    "wget",
]

a = Analysis(
    [str(repo / "voice" / "__main__.py")],
    pathex=[str(repo), str(vendor)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(hooks)],
    excludes=["FlagEmbedding", "gradio", "qdrant_client", "rank_bm25"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="voice-worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    contents_directory=".",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="voice-worker",
)
