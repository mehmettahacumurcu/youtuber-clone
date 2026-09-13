"""Generate the pinned, no-UI Applio RVC proof-of-concept Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


def _source_lines(source: str) -> list[str]:
    return source.strip("\n").splitlines(keepends=True)


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(source)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _source_lines(source),
    }


def build_notebook() -> dict:
    """Return the complete, deterministic Colab notebook JSON object."""
    eval_source = Path(__file__).with_name("rvc_eval.py").read_text(encoding="utf-8")
    cells: list[dict] = []
    cells.append(
        _markdown(
            """
# Speaker RVC POC — XTTS timbre correction

Train a supervised RVC v2 converter on the frozen 24-minute experiment bundle, then compare identical raw XTTS probes across checkpoints and retrieval rates. This is private research audio: do not publish it or represent it as a genuine statement by the target speaker.
"""
        )
    )
    cells.append(_markdown("## 1. Configuration"))
    cells.append(
        _code(
            r'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re, shutil, subprocess, sys, zipfile

RUN_FULL_DATASET = False
MODEL_NAME = "speaker_rvc_poc_v1" if not RUN_FULL_DATASET else "speaker_rvc_full_v1"
BUNDLE_NAME = "speaker_rvc_poc_24m.zip" if not RUN_FULL_DATASET else "speaker_rvc_full_74m.zip"
DRIVE_ROOT = Path("/content/drive/MyDrive/speaker_rvc")
BUNDLE_PATH = DRIVE_ROOT / BUNDLE_NAME
DRIVE_OUT = DRIVE_ROOT / MODEL_NAME
APPLIO = Path("/content/Applio")
WORK = Path("/content/speaker_rvc_work")
BUNDLE_ROOT = WORK / "bundle"
COMPARISON_DIR = DRIVE_OUT / "comparisons" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

APPLIO_COMMIT = "3e5e248c2bd003f65f3e128dd12677533def27be"
SAMPLE_RATE = 32000
TOTAL_EPOCHS = 200 if not RUN_FULL_DATASET else 250
BATCH_SIZE = 8
SAVE_EVERY_EPOCH = 25
F0_METHOD = "rmvpe"
EMBEDDER = "contentvec"
VOCODER = "HiFi-GAN"
AUDITION_EPOCHS = (100, 150, 200)
CHOSEN_EPOCH = 150
INDEX_RATES = (0.25, 0.50, 0.75)

assert not RUN_FULL_DATASET, "POC first: review the five-probe gate before enabling full training"
print({"model": MODEL_NAME, "bundle": str(BUNDLE_PATH), "epochs": TOTAL_EPOCHS, "batch": BATCH_SIZE})
'''
        )
    )

    cells.append(_markdown("## 2. Mount Drive and verify GPU"))
    cells.append(
        _code(
            r'''
from google.colab import drive
drive.mount("/content/drive")

import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. Select Runtime > Change runtime type > GPU.")
gpu = torch.cuda.get_device_properties(0)
free_gb = shutil.disk_usage("/content").free / 1e9
print("Python", sys.version)
print("Torch", torch.__version__, "CUDA", torch.version.cuda)
print("GPU", gpu.name, "VRAM_GB", round(gpu.total_memory / 1e9, 2), "disk_free_GB", round(free_gb, 1))
if free_gb < 25:
    raise RuntimeError(f"At least 25 GB ephemeral disk is required; only {free_gb:.1f} GB is free")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
'''
        )
    )

    cells.append(_markdown("## 3. Install pinned Applio"))
    cells.append(
        _code(
            r'''
def run(command, *, cwd=None):
    printable = " ".join(str(item) for item in command)
    print("+", printable, flush=True)
    subprocess.run([str(item) for item in command], cwd=cwd, check=True)

if not APPLIO.exists():
    run(["git", "clone", "--filter=blob:none", "https://github.com/IAHispano/Applio.git", APPLIO])
run(["git", "-C", APPLIO, "fetch", "--tags", "--force"])
run(["git", "-C", APPLIO, "checkout", "--detach", APPLIO_COMMIT])
# Audit equivalent: git rev-parse HEAD
actual_commit = subprocess.check_output(
    ["git", "-C", APPLIO, "rev-parse", "HEAD"], text=True
).strip()
assert actual_commit == APPLIO_COMMIT, (actual_commit, APPLIO_COMMIT)

run([sys.executable, "-m", "pip", "install", "-q", "uv"])
run([
    "uv", "pip", "install", "--system", "-q", "-r", "requirements.txt",
    "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    "--index-strategy", "unsafe-best-match",
], cwd=APPLIO)
run([
    sys.executable, "core.py", "prerequisites",
    "--models", "True", "--pretraineds_hifigan", "True",
], cwd=APPLIO)
print("Pinned Applio ready:", actual_commit)
'''
        )
    )

    cells.append(_markdown("## 4. Extract and verify experiment bundle"))
    cells.append(
        _code(
            r'''
if not BUNDLE_PATH.is_file():
    raise FileNotFoundError(f"Upload the experiment bundle to {BUNDLE_PATH}")
if BUNDLE_ROOT.exists():
    shutil.rmtree(BUNDLE_ROOT)
BUNDLE_ROOT.mkdir(parents=True)

with zipfile.ZipFile(BUNDLE_PATH) as archive:
    bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"Corrupt ZIP member: {bad_member}")
    archive.extractall(BUNDLE_ROOT)

manifest_path = BUNDLE_ROOT / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["schema"] == "rvc_poc_bundle.v1", manifest.get("schema")
for section in ("training", "eval_refs", "probes"):
    for row in manifest[section]:
        path = BUNDLE_ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row["sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {row['path']}")
if not RUN_FULL_DATASET:
    assert 1440.0 <= manifest["training_duration_s"] <= 1451.1
    assert len(manifest["eval_refs"]) == 8
    assert len(manifest["probes"]) == 5
print("Bundle verified:", len(manifest["training"]), "clips", round(manifest["training_duration_s"] / 60, 2), "min")
'''
        )
    )

    cells.append(_markdown("## 5. Bind resumable logs to Drive"))
    cells.append(
        _code(
            r'''
logs_root = APPLIO / "logs"
logs_root.mkdir(parents=True, exist_ok=True)
drive_log_dir = DRIVE_OUT / "logs" / MODEL_NAME
drive_log_dir.mkdir(parents=True, exist_ok=True)
local_log_dir = logs_root / MODEL_NAME
if local_log_dir.is_symlink():
    local_log_dir.unlink()
elif local_log_dir.exists():
    shutil.copytree(local_log_dir, drive_log_dir, dirs_exist_ok=True)
    shutil.rmtree(local_log_dir)
os.symlink(str(drive_log_dir), str(local_log_dir), target_is_directory=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
print("Resumable log directory:", drive_log_dir)
'''
        )
    )

    cells.append(_markdown("## 6. Preprocess curated audio"))
    cells.append(
        _code(
            r'''
preprocess_marker = drive_log_dir / ".preprocess_complete"
if not preprocess_marker.exists():
    run([
        sys.executable, "core.py", "preprocess",
        "--model_name", MODEL_NAME,
        "--dataset_path", str(BUNDLE_ROOT / "dataset"),
        "--sample_rate", str(SAMPLE_RATE),
        "--cpu_cores", str(os.cpu_count() or 2),
        "--cut_preprocess", "Skip",
        "--process_effects", "False",
        "--noise_reduction", "False",
        "--noise_reduction_strength", "0.7",
        "--chunk_len", "3.0",
        "--overlap_len", "0.3",
        "--normalization_mode", "none",
    ], cwd=APPLIO)
    preprocess_marker.write_text("complete\n", encoding="utf-8")
else:
    print("Preprocess already complete; using Drive state")
'''
        )
    )

    cells.append(_markdown("## 7. Extract RMVPE and ContentVec features"))
    cells.append(
        _code(
            r'''
extract_marker = drive_log_dir / ".extract_complete"
if not extract_marker.exists():
    run([
        sys.executable, "core.py", "extract",
        "--model_name", MODEL_NAME,
        "--f0_method", F0_METHOD,
        "--sample_rate", str(SAMPLE_RATE),
        "--cpu_cores", str(os.cpu_count() or 2),
        "--gpu", "0",
        "--embedder_model", EMBEDDER,
        "--embedder_model_custom", "",
        "--include_mutes", "2",
    ], cwd=APPLIO)
    extract_marker.write_text("complete\n", encoding="utf-8")
else:
    print("Feature extraction already complete; using Drive state")
'''
        )
    )

    cells.append(_markdown("## 8. Train RVC v2 and build index"))
    cells.append(
        _code(
            r'''
train_marker = drive_log_dir / f".trained_{TOTAL_EPOCHS}"
if not train_marker.exists():
    run([
        sys.executable, "core.py", "train",
        "--model_name", MODEL_NAME,
        "--save_every_epoch", str(SAVE_EVERY_EPOCH),
        "--save_only_latest", "False",
        "--save_every_weights", "True",
        "--total_epoch", str(TOTAL_EPOCHS),
        "--sample_rate", str(SAMPLE_RATE),
        "--batch_size", str(BATCH_SIZE),
        "--gpu", "0",
        "--pretrained", "True",
        "--custom_pretrained", "False",
        "--overtraining_detector", "False",
        "--cleanup", "False",
        "--cache_data_in_gpu", "False",
        "--vocoder", VOCODER,
        "--checkpointing", "False",
    ], cwd=APPLIO)
    train_marker.write_text("complete\n", encoding="utf-8")
else:
    print("Requested training ceiling already completed")
'''
        )
    )

    cells.append(_markdown("## 9. Verify and export artifacts"))
    cells.append(
        _code(
            r'''
all_weights = sorted(drive_log_dir.rglob("*.pth"))
weights = [
    path for path in all_weights
    if not path.name.startswith(("G_", "D_"))
]
indexes = sorted(drive_log_dir.rglob("*.index"))
if not weights:
    raise RuntimeError(f"No RVC inference weights found under {drive_log_dir}")
if not indexes:
    raise RuntimeError(f"No FAISS index found under {drive_log_dir}")
index_path = max(indexes, key=lambda path: path.stat().st_mtime)
artifact_dir = DRIVE_OUT / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
for path in [*weights, index_path]:
    shutil.copy2(path, artifact_dir / path.name)
print("weights:", [path.name for path in weights])
print("index:", index_path.name)
print("Training checkpoints remain in Drive logs for resume; results export contains inference weights only")
'''
        )
    )

    cells.append(_markdown("## 10. Audition epochs 100, 150, and 200"))
    cells.append(
        _code(
            r'''
from IPython.display import Audio, display

def epoch_number(path):
    matches = re.findall(r"(?:_|-)(\d+)e(?:_|-|\.)", path.name)
    return int(matches[-1]) if matches else None

epoch_weights = {epoch_number(path): path for path in weights if epoch_number(path) is not None}

def require_epoch(epoch):
    if epoch not in epoch_weights:
        available = sorted(value for value in epoch_weights if value is not None)
        raise RuntimeError(f"Requested epoch {epoch} is missing; available epochs: {available}")
    return epoch_weights[epoch]

def infer(source_path, output_path, weight_path, index_rate):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "core.py", "infer",
        "--pitch", "0",
        "--volume_envelope", "0.8",
        "--index_rate", str(index_rate),
        "--protect", "0.5",
        "--f0_autotune", "False",
        "--f0_method", F0_METHOD,
        "--input_path", str(source_path),
        "--output_path", str(output_path),
        "--pth_path", str(weight_path),
        "--index_path", str(index_path),
        "--split_audio", "False",
        "--clean_audio", "False",
        "--clean_strength", "0.7",
        "--export_format", "WAV",
        "--embedder_model", EMBEDDER,
        "--embedder_model_custom", "",
        "--formant_shifting", "False",
        "--formant_qfrency", "1.0",
        "--formant_timbre", "1.0",
        "--post_process", "False",
    ], cwd=APPLIO)
    if not output_path.is_file():
        raise RuntimeError(f"Inference did not create {output_path}")

probe_one = BUNDLE_ROOT / manifest["probes"][0]["path"]
display(Audio(str(probe_one)), "raw XTTS")
audition_rows = []
for epoch in AUDITION_EPOCHS:
    weight = require_epoch(epoch)
    output = COMPARISON_DIR / f"probe_01_e{epoch}_i050.wav"
    infer(probe_one, output, weight, 0.50)
    audition_rows.append({"probe_id": "probe_01", "epoch": epoch, "index_rate": 0.50, "path": str(output)})
    print("epoch", epoch)
    display(Audio(str(output)))
'''
        )
    )

    cells.append(_markdown("## 11. Choose epoch and sweep index rates"))
    cells.append(
        _code(
            r'''
# After listening above, edit CHOSEN_EPOCH in Configuration or here and rerun this cell.
chosen_weight = require_epoch(CHOSEN_EPOCH)
conversion_rows = []
for probe in manifest["probes"]:
    source = BUNDLE_ROOT / probe["path"]
    conversion_rows.append({
        "kind": "raw", "probe_id": probe["id"], "text": probe["text"],
        "path": str(source), "epoch": None, "index_rate": None,
    })
    for rate in INDEX_RATES:
        rate_tag = f"{round(rate * 100):03d}"
        output = COMPARISON_DIR / f"{probe['id']}_e{CHOSEN_EPOCH}_i{rate_tag}.wav"
        infer(source, output, chosen_weight, rate)
        conversion_rows.append({
            "kind": "converted", "probe_id": probe["id"], "text": probe["text"],
            "path": str(output), "epoch": CHOSEN_EPOCH, "index_rate": rate,
        })

conversion_manifest = {
    "schema": "rvc_conversion_manifest.v1",
    "model_name": MODEL_NAME,
    "checkpoint": str(chosen_weight),
    "checkpoint_sha256": hashlib.sha256(chosen_weight.read_bytes()).hexdigest(),
    "index": str(index_path),
    "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    "settings": {"pitch": 0, "f0_method": F0_METHOD, "protect": 0.5, "volume_envelope": 0.8},
    "rows": conversion_rows,
}
(COMPARISON_DIR / "conversion_manifest.json").write_text(
    json.dumps(conversion_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("comparison outputs:", len(conversion_rows), "->", COMPARISON_DIR)
'''
        )
    )

    cells.append(_markdown("## 12. Compute metrics and listening sheet"))
    cells.append(
        _code(
            r'''
import numpy as np
import soundfile as sf

raw_by_probe = {row["probe_id"]: Path(row["path"]) for row in conversion_rows if row["kind"] == "raw"}
metric_rows = []
for row in conversion_rows:
    if row["kind"] != "converted":
        continue
    source_audio, source_sr = sf.read(raw_by_probe[row["probe_id"]], dtype="float32", always_2d=False)
    output_audio, output_sr = sf.read(row["path"], dtype="float32", always_2d=False)
    if output_audio.ndim != 1 or not np.isfinite(output_audio).all():
        raise RuntimeError(f"Invalid converted audio: {row['path']}")
    duration_ratio = (len(output_audio) / output_sr) / (len(source_audio) / source_sr)
    peak = float(np.max(np.abs(output_audio))) if output_audio.size else 0.0
    clipped_fraction = float(np.mean(np.abs(output_audio) >= 0.999)) if output_audio.size else 1.0
    metric_rows.append({
        **row,
        "duration_ratio": duration_ratio,
        "duration_pass": 0.95 <= duration_ratio <= 1.05,
        "peak": peak,
        "clipped_fraction": clipped_fraction,
        "clipping_pass": clipped_fraction < 0.001,
    })

metrics = {"schema": "rvc_metrics.v1", "automatic_audio": metric_rows, "cer": None, "ecapa": None}
(COMPARISON_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
listening_sheet = [
    {"probe_id": probe["id"], "identity": "", "intelligibility": "", "artifacts": "", "preferred_index_rate": ""}
    for probe in manifest["probes"]
]
(COMPARISON_DIR / "listening_sheet.json").write_text(
    json.dumps(listening_sheet, ensure_ascii=False, indent=2), encoding="utf-8"
)
for probe in manifest["probes"]:
    print("\n", probe["id"], probe["text"], "\nraw")
    display(Audio(str(raw_by_probe[probe["id"]])))
    for row in conversion_rows:
        if row["kind"] == "converted" and row["probe_id"] == probe["id"]:
            print("index", row["index_rate"])
            display(Audio(row["path"]))
print("Basic audio metrics written. The isolated CER/ECAPA cell is added by the evaluation task.")
'''
        )
    )

    eval_cell = r'''
# Generated from training.rvc_eval; run it outside the Applio dependency environment.
eval_script = Path("/content/rvc_eval.py")
eval_script.write_text(__EVAL_SOURCE__, encoding="utf-8")
eval_env = Path("/content/rvc-eval-env")
eval_python = eval_env / "bin" / "python"
full_metrics = COMPARISON_DIR / "metrics_full.json"
conversion_manifest_path = COMPARISON_DIR / "conversion_manifest.json"
try:
    if not eval_python.is_file():
        run(["uv", "venv", "--system-site-packages", str(eval_env)])
    run([
        "uv", "pip", "install", "--python", str(eval_python),
        "speechbrain<1.0", "faster-whisper>=1.2.1",
    ])
    run([
        str(eval_python), str(eval_script),
        "--bundle-root", str(BUNDLE_ROOT),
        "--conversion-manifest", str(conversion_manifest_path),
        "--output", str(full_metrics),
        "--cache-dir", "/content/rvc_eval_cache",
        "--allow-model-failure",
    ])
    print("Full CER/ECAPA report:", full_metrics)
except subprocess.CalledProcessError as error:
    fallback = {
        "error": str(error),
        "audio_metrics_preserved": str(COMPARISON_DIR / "metrics.json"),
    }
    (COMPARISON_DIR / "evaluation_setup_error.json").write_text(
        json.dumps(fallback, indent=2), encoding="utf-8"
    )
    print("LOCAL EVALUATION FALLBACK:")
    print("python -m training.rvc_eval --bundle-root <bundle> --conversion-manifest <conversion_manifest.json> --output <metrics_full.json> --allow-model-failure")
'''.replace("__EVAL_SOURCE__", json.dumps(eval_source, ensure_ascii=False))
    cells.append(_code(eval_cell))

    cells.append(_markdown("## 13. Export results ZIP"))
    cells.append(
        _code(
            r'''
training_manifest = {
    "applio_commit": APPLIO_COMMIT,
    "model_name": MODEL_NAME,
    "sample_rate": SAMPLE_RATE,
    "epochs": TOTAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "save_every_epoch": SAVE_EVERY_EPOCH,
    "f0_method": F0_METHOD,
    "embedder": EMBEDDER,
    "vocoder": VOCODER,
    "bundle_sha256": hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest(),
}
(DRIVE_OUT / "training_manifest.json").write_text(
    json.dumps(training_manifest, indent=2), encoding="utf-8"
)
shutil.copy2(BUNDLE_ROOT / "manifest.json", DRIVE_OUT / "bundle_manifest.json")
result_zip = DRIVE_OUT / "speaker_rvc_poc_v1_results.zip"
with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as archive:
    for root in (artifact_dir, COMPARISON_DIR):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(DRIVE_OUT).as_posix())
    for name in ("training_manifest.json", "bundle_manifest.json"):
        archive.write(DRIVE_OUT / name, name)
bad_member = zipfile.ZipFile(result_zip).testzip()
assert bad_member is None, bad_member
print("RESULT READY:", result_zip, "size_GB", round(result_zip.stat().st_size / 1e9, 2))
'''
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path | None = None) -> Path:
    """Write the generated notebook as UTF-8 JSON and return its path."""
    output = Path(path) if path is not None else Path(__file__).with_name(
        "rvc_finetune_colab.ipynb"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return output


def main() -> None:
    output = write_notebook()
    print(f"wrote {output} ({len(build_notebook()['cells'])} cells)")


if __name__ == "__main__":
    main()
