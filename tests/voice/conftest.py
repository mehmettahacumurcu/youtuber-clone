from __future__ import annotations

import builtins

import pytest


@pytest.fixture
def forbid_torch_import(monkeypatch):
    """Fail if the exercised fake-backend path attempts a Torch import."""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"unexpected eager Torch import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
