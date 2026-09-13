"""Derive training/qwen_v5_export.ipynb from qwen_v5_train.ipynb — EXPORT-ONLY notebook.

Reads qwen_v5_train.ipynb, comments out every training/probe cell, prepends
"disabled" notes to their section headings, pre-fills Section 10's WINNERS with the
7 export picks, bumps the Modelfile num_ctx to 8192, and adds skip-if-done guards —
so Runtime -> Run all on A100/L4 produces 7 Q4_K_M GGUFs + Modelfiles and nothing else.

Never modifies qwen_v5_train.ipynb.

Run:  .venv\\Scripts\\python.exe -m training.build_v5_export_nb
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "qwen_v5_train.ipynb"
DST = HERE / "qwen_v5_export.ipynb"

# ── New markdown header inserted at index 0 (EXACT content) ───────────────────────────────────────
HEADER_MD = r'''# EXPORT PREP — run-all produces 7 GGUFs (no training)

This is `qwen_v5_train.ipynb` prepped for **export only**: every training/probe cell below is commented out, so **Runtime → Run all** executes just Install → Auth+Drive → Config → Section 10 and writes **7 Q4_K_M GGUFs + Modelfiles** to `MyDrive/speaker_models/`.

**Runtime:** A100 (best) or L4. Not T4 — the ≥22 GB VRAM assert in Section 2 will stop you, and the 14B merge wants the bigger system RAM anyway.

**WINNERS (7)** — ranked by the 2026-07-02/03 read of `speaker_v5_outputs.md`; top-5 were all Arm A, so Arm B's best 2 are appended per the selection rule:

| rank | ckpt | why |
|---|---|---|
| 1 | A_636 | sharpest grounded answers; clean in-voice declines |
| 2 | A_742 | best declines on fact-baits |
| 3 | A_848 | abstention present; most coherent late ckpt; fact-accurate |
| 4 | A_530 | abstention onset; strong voice, slightly more noise |
| 5 | A_424 | good voice; pre-abstention |
| +B1 | B_742 | most fluent Arm B; fact-accurate (confabulates names — known B trait) |
| +B2 | B_636 | only 20/20 clean stops in the run; solid voice |

**Changes vs the original Section 10:**
- `WINNERS` pre-filled with the 7 picks above.
- Modelfiles write `num_ctx 8192` (was 4096) to match the local run2-ep2 baseline Modelfile for a fair A/B.
- Skip-if-done: a winner whose `.gguf` already exists on Drive is skipped, so a dead session resumes cheaply on re-run.

**Budget:** ~20–35 min per checkpoint after one-time setup (~3–4 h total). Drive needs ~60 GB free (7 × ~8.5 GB) — trim `WINNERS` in Section 10 if tight. If the VM disk fills before the two B exports (both base models cache ~28 GB each + ~56 GB peak intermediates), clear the Arm-A base cache first: `!rm -rf ~/.cache/huggingface/hub/models--huihui-ai--*`

**Locally afterwards** (per model, from its own folder under `models\`, mirroring `models\qwen3-run2-ep2\`):
```
ollama create speaker-v5-a636 -f Modelfile-speaker-v5-A-step0636
```
The studio lists new Ollama models automatically.'''

# ── Comment banner (prepended to each disabled code cell) ─────────────────────────────────────────
BANNER = "# [EXPORT PREP] disabled — not needed for GGUF export\n"

# ── Prefix prepended to each disabled section's markdown heading ──────────────────────────────────
MD_PREFIX = "> **[export prep]** Section disabled — not needed for GGUF export.\n\n"

# ── Section-10 export-cell edits ──────────────────────────────────────────────────────────────────
OLD_A = 'WINNERS = []   # e.g. [("A", 424), ("B", 530)] — fill from the outputs.md read, then run'
NEW_A = """WINNERS = [        # ranked by the 2026-07-03 read; top-5 all Arm A -> B's best 2 appended per rule
    ("A", 636),    # 1: sharpest grounded, clean declines
    ("A", 742),    # 2: best declines
    ("A", 848),    # 3: abstention + most coherent late ckpt, fact-accurate
    ("A", 530),    # 4: abstention onset, strong voice
    ("A", 424),    # 5: good voice, pre-abstention
    ("B", 742),    # B1: most fluent B, fact-accurate
    ("B", 636),    # B2: only 20/20 clean stops, solid voice
]"""

OLD_B = '"PARAMETER num_ctx 4096",'
NEW_B = '"PARAMETER num_ctx 8192",   # match run2-ep2 baseline Modelfile for a fair A/B'

OLD_C = r'''    assert os.path.isdir(_adir), f"checkpoint not found: {_adir}"
    print("=" * 72, f"\nEXPORT Arm {arm} step {step:04d}")'''
NEW_C = r'''    assert os.path.isdir(_adir), f"checkpoint not found: {_adir}"
    _gguf_name = f"speaker-qwen3-14b-v5-{arm}-step{step:04d}-q4_k_m.gguf"
    if os.path.exists(f"{OUTPUT_ROOT}/{_gguf_name}"):
        print(f"SKIP Arm {arm} step {step:04d} — {_gguf_name} already on Drive")
        return
    print("=" * 72, f"\nEXPORT Arm {arm} step {step:04d}")'''


def comment_out(src: str) -> str:
    """Banner line + every original line prefixed with '# ' (keeps line count)."""
    lines = src.split("\n")
    return BANNER + "\n".join("# " + ln for ln in lines)


def replace_once(src: str, old: str, new: str, ctx: str) -> str:
    n = src.count(old)
    assert n == 1, f"{ctx}: expected exactly 1 occurrence of {old!r}, found {n}"
    return src.replace(old, new)


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, f"expected nbformat 4 in source, got {nb.get('nbformat')!r}"
    cells = nb["cells"]
    n_orig = len(cells)

    # Normalize every source (list-of-lines or string) to a single string.
    for c in cells:
        s = c.get("source", "")
        c["source"] = "".join(s) if isinstance(s, list) else s

    keep_ids: set[int] = set()

    def one_code(needle: str, in_keep: bool = False) -> dict:
        # Search code cells. COMMENT/export targets exclude the 4 KEEP cells so that a
        # substring incidentally present in a kept cell (e.g. 'aya_dataset', which also
        # appears in an auth-cell comment) still resolves to exactly one eligible cell.
        hits = [c for c in cells if c["cell_type"] == "code" and needle in c["source"]
                and (in_keep or id(c) not in keep_ids)]
        assert len(hits) == 1, \
            f"substring {needle!r}: expected exactly 1 code cell, found {len(hits)}"
        return hits[0]

    def one_md(needle: str) -> dict:
        hits = [c for c in cells if c["cell_type"] == "markdown" and needle in c["source"]]
        assert len(hits) == 1, \
            f"substring {needle!r}: expected exactly 1 markdown cell, found {len(hits)}"
        return hits[0]

    # 2. Keep these code cells ACTIVE + byte-identical — assert each is present, don't touch,
    #    and record their identity so the COMMENT/export searches never re-select them.
    KEEP = ['!pip install unsloth', 'print("unsloth"', 'drive.mount', 'BASE_A = ']
    for needle in KEEP:
        keep_ids.add(id(one_code(needle, in_keep=True)))
    assert len(keep_ids) == 4, f"KEEP cells collided — expected 4 distinct, got {len(keep_ids)}"

    # 3. Comment out these 9 code cells.
    COMMENT = [
        'TRAIN = _find(',      # data upload
        '_steps_per_epoch',    # step math
        'def load_fresh_arm',  # helpers
        'aya_dataset',         # replay construction
        'tr_A.train()',        # Arm A training
        'tr_B.train()',        # Arm B training
        'PROBES = [',          # probe list literal
        'def run_probes_on',   # generation pass
        '_OUT_MD',             # write outputs
    ]
    commented = 0
    for needle in COMMENT:
        c = one_code(needle)
        c["source"] = comment_out(c["source"])
        commented += 1

    # 4. Prepend the disabled-section note to these 7 markdown headings.
    MD_SECTIONS = [
        '## 3. Data', '## 5. Helpers', '### 5a. Replay', '## 6. Arm A',
        '## 7. Arm B', '## 8. Generate', '## 9. Write output',
    ]
    prepended = 0
    for needle in MD_SECTIONS:
        c = one_md(needle)
        c["source"] = MD_PREFIX + c["source"]
        prepended += 1

    # 5. Modify the Section-10 export cell (3 exact, single-occurrence replacements).
    exp = one_code('WINNERS = []')
    s = exp["source"]
    s = replace_once(s, OLD_A, NEW_A, "export WINNERS list")
    s = replace_once(s, OLD_B, NEW_B, "export num_ctx bump")
    s = replace_once(s, OLD_C, NEW_C, "export skip-if-done block")
    exp["source"] = s

    # 1. Insert the new markdown header at index 0.
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": HEADER_MD})

    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

    kept = n_orig - commented - prepended - 1  # minus the 1 export cell that was modified
    print(f"wrote {DST}")
    print(f"  cells total     : {len(cells)}  (was {n_orig} + 1 inserted markdown header)")
    print(f"  kept unchanged  : {kept}")
    print(f"  commented out   : {commented}")
    print(f"  prepended md    : {prepended}")
    print(f"  export modified : 1")
    print(f"  inserted header : 1")


if __name__ == "__main__":
    main()
