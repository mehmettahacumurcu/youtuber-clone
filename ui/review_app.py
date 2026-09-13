"""Gradio review app — reference pool approval + borderline segment review + status.

Launch with:
    uv run python -m ui.review_app

Three tabs:
    1. References  -> approve/reject candidates from data/reference/pool/candidates.json
    2. Borderline  -> manual keep/drop/mixed votes on segments with sim in
                     the borderline band, persisted to data/decisions.sqlite
    3. Status      -> per-video stage completion + counts

The reference pool changes here directly affect Stage 4 once re-run; the
borderline decisions feed Stage 6 (filter) downstream.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from pipeline.config import REPO_ROOT, load_settings
from pipeline.download import audio_path_for
from pipeline.identify.pool import candidate_embeddings_path, candidates_manifest_path
from ui.state import (
    active_pool_path,
    decision_counts,
    get_decision,
    load_active_pool,
    record_decision,
    save_active_pool,
)

log = logging.getLogger("ui.review_app")

SETTINGS = load_settings()


# ---------- references tab ----------


def _load_candidates() -> list[dict[str, Any]]:
    p = candidates_manifest_path(SETTINGS)
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("candidates", [])


def _load_candidate_embeddings():
    """Return numpy array (N, D) of L2-normalized ECAPA embeddings, or None if missing."""
    import numpy as np

    p = candidate_embeddings_path(SETTINGS)
    if not p.exists():
        return None
    return np.load(str(p))


def _current_active_set() -> set[int]:
    a = load_active_pool(SETTINGS)
    return set(a.get("active_indices", []))


def _compute_row_states(
    candidates: list[dict],
    embeddings,  # np.ndarray (N, D) or None
    active: list[int],
    k_target: int,
) -> tuple[list[dict], str]:
    """Return per-candidate {'state': 'active'|'proposed'|'extra', 'score': float} list + status text.

    'score' is min cosine distance from this candidate to any active member
    (higher = more diverse vs current pool).
    """
    import numpy as np

    n = len(candidates)
    active_set = set(active)
    if embeddings is None or n == 0:
        states = [{"state": "active" if c["candidate_index"] in active_set else "extra", "score": 0.0}
                  for c in candidates]
        return states, f"Active: {len(active_set)} / {k_target} — (no embeddings)"

    # candidate_index assumed equal to position in embeddings array
    if active_set:
        act_emb = embeddings[sorted(active_set)]  # (A, D)
        # cos(emb, a) = emb @ a.T (both already normalized)
        sims = embeddings @ act_emb.T  # (N, A)
        max_sim_to_active = sims.max(axis=1)  # (N,)
        min_dist_to_active = 1.0 - max_sim_to_active  # higher = farther from any active
    else:
        # No active yet → "distance" is just 1 - sim_to_seed roughly
        min_dist_to_active = np.ones(n, dtype=np.float32)

    non_active_idxs = [i for i in range(n) if i not in active_set]
    non_active_idxs.sort(key=lambda i: -float(min_dist_to_active[i]))
    proposed_set = set(non_active_idxs[: max(0, k_target - len(active_set))])

    states = []
    for i in range(n):
        if i in active_set:
            state = "active"
        elif i in proposed_set:
            state = "proposed"
        else:
            state = "extra"
        states.append({"state": state, "score": float(min_dist_to_active[i])})

    status = f"Active: {len(active_set)} / {k_target}"
    if len(active_set) < k_target:
        status += f" — propose {k_target - len(active_set)} more (sorted by diversity below)"
    else:
        status += " — target reached. Save when ready."
    return states, status


def _row_button_props(state: str, score: float) -> tuple[str, str]:
    """Return (button_label, button_variant) for a row state."""
    if state == "active":
        return "Remove", "secondary"
    if state == "proposed":
        return f"Approve  ★ div={score:.3f}", "primary"
    # extra (below k cutoff)
    return f"Approve  (div={score:.3f})", "secondary"


def _row_info_md(c: dict, state: str, score: float) -> str:
    label = "SEED" if c.get("is_seed") else (c.get("source_speaker_label") or "?")
    badge = {"active": "✅ ACTIVE", "proposed": "🟢 PROPOSED", "extra": "⚪ extra"}[state]
    return (
        f"### #{c['candidate_index']:02d} — {badge}\n"
        f"**source:** `{c['source_video']}` @ {c['start']:.1f}–{c['end']:.1f}s  ({c['duration_s']:.1f}s)  \n"
        f"**speaker label:** {label}  \n"
        f"**sim to seed:** `{c['sim_to_seed']:.3f}`  ·  "
        f"**diversity vs active pool:** `{score:.3f}`"
    )


def _build_references_tab() -> gr.Blocks:
    """References tab with K-target slider + dynamic Active/Proposed/Extra grouping."""
    candidates = _load_candidates()
    embeddings = _load_candidate_embeddings()
    initial_active = sorted(_current_active_set())
    initial_k = 20

    initial_states, initial_status = _compute_row_states(
        candidates, embeddings, initial_active, initial_k
    )

    with gr.Blocks() as tab:
        gr.Markdown(
            f"### Reference pool — {len(candidates)} candidates\n"
            "**Goal:** pick K diverse refs of *him* alone. The system proposes the next "
            "most-diverse candidate vs your current active pool (higher score = more acoustically "
            "different from refs you've already approved). Each toggle re-ranks instantly.\n\n"
            "**🟢 PROPOSED** = next-K-target picks by diversity.  "
            "**✅ ACTIVE** = currently in pool.  "
            "**⚪ extra** = above K-target rank; still approvable if you want >K refs."
        )
        with gr.Row():
            k_slider = gr.Slider(
                minimum=1, maximum=max(30, len(candidates)),
                value=initial_k, step=1, label="K (target pool size)",
            )
            count_md = gr.Markdown(initial_status)
        with gr.Row():
            save_btn = gr.Button("💾 Save active pool", variant="primary")
            status_md = gr.Markdown("")

        active_state = gr.State(initial_active)

        # Pre-render N rows in candidate_index order. State + visibility is computed dynamically.
        row_buttons: list[gr.Button] = []
        row_infos: list[gr.Markdown] = []
        for i, c in enumerate(candidates):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=200):
                    audio_path = REPO_ROOT / c["file"]
                    gr.Audio(
                        value=str(audio_path) if audio_path.exists() else None,
                        label=f"#{c['candidate_index']:02d}",
                        type="filepath",
                        interactive=False,
                        show_download_button=False,
                    )
                with gr.Column(scale=4):
                    info_md = gr.Markdown(
                        _row_info_md(c, initial_states[i]["state"], initial_states[i]["score"])
                    )
                with gr.Column(scale=1, min_width=160):
                    btn_label, btn_variant = _row_button_props(
                        initial_states[i]["state"], initial_states[i]["score"]
                    )
                    btn = gr.Button(btn_label, variant=btn_variant, size="sm")
                row_buttons.append(btn)
                row_infos.append(info_md)

        def _rerender(active_list: list[int], k: int) -> list:
            """Recompute all row states and return Gradio updates."""
            states, status = _compute_row_states(candidates, embeddings, active_list, k)
            btn_updates = []
            info_updates = []
            for i, c in enumerate(candidates):
                lbl, var = _row_button_props(states[i]["state"], states[i]["score"])
                btn_updates.append(gr.update(value=lbl, variant=var))
                info_updates.append(gr.update(value=_row_info_md(c, states[i]["state"], states[i]["score"])))
            return [status, *btn_updates, *info_updates]

        def _on_k_change(active_list: list[int], k: int):
            return _rerender(active_list, int(k))

        k_slider.change(
            _on_k_change,
            inputs=[active_state, k_slider],
            outputs=[count_md, *row_buttons, *row_infos],
        )

        # Per-row toggle handlers — each baked with its own candidate index via closure.
        for slot_i, c in enumerate(candidates):
            cand_idx = c["candidate_index"]

            def _make_toggle(ci: int):
                def _toggle(active_list: list[int], k: int):
                    aset = set(active_list)
                    if ci in aset:
                        aset.remove(ci)
                    else:
                        aset.add(ci)
                    new_active = sorted(aset)
                    re = _rerender(new_active, int(k))
                    # Prepend the new active_state for the State output
                    return [new_active, *re]
                return _toggle

            row_buttons[slot_i].click(
                _make_toggle(cand_idx),
                inputs=[active_state, k_slider],
                outputs=[active_state, count_md, *row_buttons, *row_infos],
            )

        def _save(active_list: list[int]) -> str:
            p = save_active_pool(SETTINGS, active_list)
            return (
                f"✅ Saved **{len(active_list)}** active references to "
                f"`{p.relative_to(REPO_ROOT)}`. "
                "Tell Claude to re-run Stage 4 to pick up the new pool."
            )

        save_btn.click(_save, inputs=[active_state], outputs=[status_md])

        # On page (re)load, refresh state from disk so browser F5 doesn't show
        # stale "0 / K" when active.json on disk already has saved entries.
        def _refresh_on_load(k: int):
            from_disk = sorted(_current_active_set())
            re = _rerender(from_disk, int(k))
            return [from_disk, *re]

        tab.load(
            _refresh_on_load,
            inputs=[k_slider],
            outputs=[active_state, count_md, *row_buttons, *row_infos],
        )
    return tab


# ---------- borderline review tab ----------


def _list_borderline_segments(
    sim_lo: float, sim_hi: float, only_undecided: bool = True
) -> list[dict[str, Any]]:
    """Walk identify/*.json and return segments whose similarity is in [sim_lo, sim_hi)."""
    out = []
    for jp in sorted(SETTINGS.paths.identify_dir.glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        vid = d["video_id"]
        for s in d["segments"]:
            sim = s.get("similarity")
            if sim is None:
                continue
            if not (sim_lo <= sim < sim_hi):
                continue
            existing = (
                get_decision(SETTINGS, video_id=vid, start=s["start"], end=s["end"])
                if only_undecided
                else None
            )
            if only_undecided and existing is not None:
                continue
            out.append(
                {
                    "video_id": vid,
                    "start": s["start"],
                    "end": s["end"],
                    "speaker": s["speaker"],
                    "similarity": sim,
                    "existing_decision": existing,
                }
            )
    # Sort: descending similarity (the "most likely him among the borderline" first)
    out.sort(key=lambda x: -x["similarity"])
    return out


def _slice_to_tempfile(video_id: str, start: float, end: float, context_s: float = 5.0) -> Path:
    """ffmpeg-slice a window of the source wav with ±context_s seconds around the segment."""
    src = audio_path_for(video_id, SETTINGS)
    cut_start = max(0.0, start - context_s)
    cut_end = end + context_s
    fd, tmp = tempfile.mkstemp(prefix=f"{video_id}_{start:.1f}_", suffix=".wav")
    Path(tmp).unlink()  # ffmpeg will create
    import os

    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(src),
        "-ss",
        f"{cut_start:.3f}",
        "-to",
        f"{cut_end:.3f}",
        "-ac",
        "1",
        "-ar",
        str(SETTINGS.download.sample_rate_hz),
        "-c:a",
        "pcm_s16le",
        tmp,
    ]
    subprocess.run(cmd, check=True)
    return Path(tmp)


def _build_borderline_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown(
            "### Borderline segment review\n"
            "Segments with similarity in the borderline range — listen with ±5s context "
            "and vote. Decisions persist to `data/decisions.sqlite`."
        )
        with gr.Row():
            sim_lo = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.40,
                step=0.01,
                label="sim ≥",
            )
            sim_hi = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.55,
                step=0.01,
                label="sim <",
            )
            only_undecided = gr.Checkbox(value=True, label="Only undecided")
            refresh_btn = gr.Button("Refresh queue")

        queue_state = gr.State([])
        index_state = gr.State(0)
        queue_size_md = gr.Markdown()
        position_md = gr.Markdown()
        seg_info_md = gr.Markdown()
        audio_player = gr.Audio(label="±5s context", type="filepath", interactive=False)

        with gr.Row():
            keep_btn = gr.Button("Keep (him)", variant="primary")
            drop_btn = gr.Button("Drop (not him)")
            mixed_btn = gr.Button("Mixed / overlapping")
            skip_btn = gr.Button("Skip")

        def _refresh(lo: float, hi: float, undec: bool) -> tuple[list[dict], int, str, str, str, Any]:
            q = _list_borderline_segments(lo, hi, only_undecided=undec)
            size_md = f"**{len(q)} segments** in queue (sim {lo:.2f} ≤ s < {hi:.2f}{', undecided only' if undec else ''})"
            if not q:
                return q, 0, size_md, "—", "queue empty.", None
            s = q[0]
            pos_md = f"**1 / {len(q)}**"
            info_md = (
                f"`{s['video_id']}` @ {s['start']:.2f}–{s['end']:.2f}s  "
                f"(speaker `{s['speaker']}`, sim `{s['similarity']:.3f}`)"
            )
            tmp = _slice_to_tempfile(s["video_id"], s["start"], s["end"])
            return q, 0, size_md, pos_md, info_md, str(tmp)

        def _vote(
            decision: str,
            q: list[dict],
            i: int,
        ) -> tuple[int, str, str, Any]:
            if not q or i >= len(q):
                return i, "queue empty.", "—", None
            s = q[i]
            record_decision(
                SETTINGS,
                video_id=s["video_id"],
                start=s["start"],
                end=s["end"],
                similarity=s["similarity"],
                speaker=s["speaker"],
                decision=decision,
            )
            i_next = i + 1
            if i_next >= len(q):
                return i_next, f"**{len(q)} / {len(q)}** — done.", "queue complete.", None
            s_next = q[i_next]
            pos = f"**{i_next + 1} / {len(q)}**"
            info = (
                f"`{s_next['video_id']}` @ {s_next['start']:.2f}–{s_next['end']:.2f}s  "
                f"(speaker `{s_next['speaker']}`, sim `{s_next['similarity']:.3f}`)"
            )
            tmp = _slice_to_tempfile(s_next["video_id"], s_next["start"], s_next["end"])
            return i_next, pos, info, str(tmp)

        refresh_btn.click(
            _refresh,
            inputs=[sim_lo, sim_hi, only_undecided],
            outputs=[queue_state, index_state, queue_size_md, position_md, seg_info_md, audio_player],
        )
        for btn, dec in [
            (keep_btn, "keep"),
            (drop_btn, "drop"),
            (mixed_btn, "mixed"),
            (skip_btn, "skip"),
        ]:
            btn.click(
                lambda q, i, d=dec: _vote(d, q, i),
                inputs=[queue_state, index_state],
                outputs=[index_state, position_md, seg_info_md, audio_player],
            )
    return tab


# ---------- status tab ----------


def _stage_status_table() -> str:
    """Render a markdown table of per-video stage completion + key stats."""
    audio_ids = sorted(p.stem for p in SETTINGS.paths.audio_dir.glob("*.wav"))
    if not audio_ids:
        return "_no videos in `data/audio/`._"

    counts = decision_counts(SETTINGS)
    active = _current_active_set()

    lines = [
        "| video | dur (min) | VAD | diarize | identify | kept speech (min) | speakers |",
        "|---|---:|:---:|:---:|:---:|---:|---:|",
    ]
    total_kept_s = 0.0
    for vid in audio_ids:
        dur_m = "?"
        try:
            import soundfile as sf

            info = sf.info(str(audio_path_for(vid, SETTINGS)))
            dur_m = f"{info.frames / info.samplerate / 60:.1f}"
        except Exception:
            pass

        vad_ok = (SETTINGS.paths.vad_dir / f"{vid}.json").exists()
        diar_path = SETTINGS.paths.diarize_dir / f"{vid}.json"
        idn_path = SETTINGS.paths.identify_dir / f"{vid}.json"
        diar_ok = diar_path.exists()
        idn_ok = idn_path.exists()

        kept_m = "—"
        n_spk = "—"
        if idn_ok:
            d = json.loads(idn_path.read_text(encoding="utf-8"))
            kept = d["summary"]["kept_speech_s"]
            total_kept_s += kept
            kept_m = f"{kept / 60:.1f}"
        if diar_ok:
            d = json.loads(diar_path.read_text(encoding="utf-8"))
            n_spk = str(d["num_speakers"])

        lines.append(
            f"| `{vid}` | {dur_m} | "
            f"{'✓' if vad_ok else '·'} | "
            f"{'✓' if diar_ok else '·'} | "
            f"{'✓' if idn_ok else '·'} | "
            f"{kept_m} | {n_spk} |"
        )

    lines.append("")
    lines.append(f"**Total kept speech (his-only):** {total_kept_s / 60:.1f} min across {len(audio_ids)} videos.")
    lines.append("")
    lines.append(f"**Active reference pool:** {len(active)} candidates ({sorted(active)})")
    if counts:
        cstr = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"**Borderline decisions logged:** {cstr}")
    else:
        lines.append("**Borderline decisions logged:** none yet")
    return "\n".join(lines)


def _build_status_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("### Pipeline status")
        status_md = gr.Markdown(_stage_status_table())
        refresh = gr.Button("Refresh")
        refresh.click(lambda: _stage_status_table(), outputs=status_md)
    return tab


# ---------- assemble ----------


def build_app() -> gr.Blocks:
    with gr.Blocks(title="youtuber-clone review") as app:
        gr.Markdown("# youtuber-clone — review app")
        with gr.Tabs():
            with gr.TabItem("References"):
                _build_references_tab()
            with gr.TabItem("Borderline review"):
                _build_borderline_tab()
            with gr.TabItem("Status"):
                _build_status_tab()
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = build_app()
    # Bind to localhost only — no external exposure. Opens default browser on start.
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, share=False)


if __name__ == "__main__":
    main()
