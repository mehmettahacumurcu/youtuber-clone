from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace


class _Component:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def click(self, *args, **kwargs):
        return self

    def then(self, *args, **kwargs):
        return self

    def submit(self, *args, **kwargs):
        return self

    def change(self, *args, **kwargs):
        return self

    def load(self, *args, **kwargs):
        return self


def _fake_gradio() -> SimpleNamespace:
    return SimpleNamespace(
        Blocks=lambda *args, **kwargs: _Component(),
        Row=lambda *args, **kwargs: _Component(),
        Column=lambda *args, **kwargs: _Component(),
        Accordion=lambda *args, **kwargs: _Component(),
        HTML=lambda *args, **kwargs: _Component(),
        Chatbot=lambda *args, **kwargs: _Component(),
        Textbox=lambda *args, **kwargs: _Component(),
        Button=lambda *args, **kwargs: _Component(),
        Dropdown=lambda *args, **kwargs: _Component(),
        Slider=lambda *args, **kwargs: _Component(),
        Audio=lambda *args, **kwargs: _Component(),
        Markdown=lambda *args, **kwargs: _Component(),
        update=lambda **kwargs: kwargs,
        themes=SimpleNamespace(Base=lambda: object()),
    )


def test_studio_import_defers_torch_to_the_real_xtts_backend(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("Studio imported torch eagerly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "gradio", _fake_gradio())
    monkeypatch.delitem(sys.modules, "ui.speaker_studio", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    importlib.import_module("ui.speaker_studio")
