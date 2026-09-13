# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


repo = Path(SPECPATH).resolve().parents[1]
hooks = repo / "packaging" / "pyinstaller" / "hooks"
datas = (
    collect_data_files("gradio", include_py_files=True)
    + collect_data_files("gradio_client")
    + collect_data_files("safehttpx")
    + collect_data_files("groovy")
    + [(str(repo / "config.yaml"), "runtime")]
)
hiddenimports = [
    "ui.speaker_studio",
    "ui.voice_client",
    "rag.retrieve_server",
    "rag.readiness",
    "pipeline.config",
    "FlagEmbedding.inference.embedder.encoder_only.base",
    "FlagEmbedding.inference.embedder.encoder_only.m3",
    "FlagEmbedding.inference.reranker.encoder_only.base",
    "transformers.models.metaclip_2",
]

a = Analysis(
    [str(repo / "ui" / "__main__.py")],
    pathex=[str(repo)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(hooks)],
    excludes=["TTS", "faiss", "torchaudio"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="studio-worker",
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
    name="studio-worker",
)
