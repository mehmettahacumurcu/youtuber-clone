"""Audited Coqui XTTS inference imports and non-model package data."""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


hiddenimports = [
    "TTS.tts.configs.xtts_config",
    "TTS.tts.layers.xtts.dvae",
    "TTS.tts.layers.xtts.gpt",
    "TTS.tts.layers.xtts.hifigan_decoder",
    "TTS.tts.layers.xtts.perceiver_encoder",
    "TTS.tts.layers.xtts.stream_generator",
    "TTS.tts.layers.xtts.tokenizer",
    "TTS.tts.models.xtts",
] + collect_submodules("TTS.vocoder.configs")

def _runtime_data(item):
    source = Path(item[0])
    parts = {part.casefold() for part in source.parts}
    return not parts.intersection({"demos", "notebooks", "test", "tests", "testing", "test_data"}) and source.suffix.casefold() not in {
        ".ipynb", ".ckpt", ".gguf", ".index", ".pth", ".pt", ".safetensors",
    }


datas = [item for item in collect_data_files("TTS", include_py_files=True) if _runtime_data(item)]
datas += [
    item
    for item in collect_data_files("TTS.vocoder.configs", include_py_files=True)
    if _runtime_data(item)
]
datas += [
    item
    for item in collect_data_files("inflect", include_py_files=True)
    if _runtime_data(item)
]
datas += [
    item
    for item in collect_data_files("gruut", include_py_files=False)
    if _runtime_data(item)
]
