# Packaged worker integration gate

This gate exercises the frozen Windows Studio and Voice executables. It does not
substitute imported Python worker modules for the packaged processes.

## Prerequisites

- Windows with the two onedir bundles built under `dist/workers/`.
- The distribution development environment at `.venv-dist-dev`.
- For the real gate, a running local Ollama service on `127.0.0.1:11434` with
  `speaker-v5-a636:latest` and `qwen3:4b` available.
- For the real voice gate, the approved RAG, XTTS, reference WAV, epoch-200 RVC,
  ContentVec, and RMVPE assets below the supplied `--real-assets-root`.

Build or reseal both components with:

```powershell
& powershell.exe -NoProfile -File scripts\build_workers.ps1
```

After a Studio-only packaging fix, `-StudioOnly` rebuilds and reseals Studio
without rebuilding the large Voice tree.

## Fixture gate

```powershell
& .venv-dist-dev\Scripts\python.exe scripts\smoke_packaged_workers.py --fixture
& .venv-dist-dev\Scripts\python.exe -m pytest tests\e2e\test_packaged_workers.py -q -p no:cacheprovider
```

Fixture mode launches both real EXEs with minimal temporary manifests and fake
loopback collaborators. It verifies live/auth boundaries, frozen RAG dispatch,
cookie attributes, Voice request ordering, unload behavior, and teardown. It
does not claim that real retrieval or synthesis succeeded.

## Real RAG and voice gate

```powershell
& .venv-dist-dev\Scripts\python.exe scripts\smoke_packaged_workers.py `
  --real-assets-root E:\youtuber-clone `
  --question "Mutlak butlan hakkında ne düşünüyorsun?" `
  --voice
```

Omit `--voice` to stop after real retrieval and Speaker-model answer generation.
Add `--keep-output` only when the staged install/data/cache trees are needed for
diagnosis. The controller otherwise removes staged private assets after the run.
It discovers source assets read-only, hashes every selected file, uses same-volume
hard links only for files consumed read-only, and never writes into either sealed
worker component. Writable Qdrant/manifest inputs are copied. The controller does
not use `chmod` on NTFS hardlinks (attributes affect the shared file record); it
records the read-only consumer contract and rehashes every hardlinked source after
worker shutdown. Any source change fails the gate. Every raw source path component
is checked before resolution or traversal, all selections must remain below the
explicit approved source root, and symlinks, junctions, and other reparse points
fail before an external target can be visited.

Every run allocates three distinct loopback ports and a new random session secret.
Children receive explicit absolute install/data/cache roots, ports,
`PYTHONNOUSERSITE=1`, and the low-VRAM runtime settings. The controller never
stops or replaces the Ollama service. Required models may be transiently
unloaded/reloaded for the low-VRAM sequence; cleanup restores their exact initial
residency and verifies that unrelated identities are unchanged.

Answer generation is submitted to the frozen Studio process through its strict
cookie-authenticated `POST /v1/chat` API in `grounded_strict` mode. The response
reports the verifier status and the count of evidence spans actually placed into
that generation prompt, plus a separate structured generation outcome. A
successful response must report `generation_status: generated`; empty output,
model errors, policy refusals, retrieval errors, and empty input never cross the
successful API schema. Frozen Studio and the external controller also reject the
exact stock empty-output, RAG-error, and policy-refusal strings even if another
layer incorrectly labels one as generated. The gate requires
`answerable|partial` plus a positive same-generation evidence count; the separate
direct retrieval is labeled only as a preflight. Any non-generated,
unsupported/error, or stock operational response fails before
`results/answer.json` is written or Voice starts. The external controller does
not import or call `chat_turn`.

## Expected outputs

The default contained output root is `build/packaged-smoke/`:

- `result.json`: structured pass/failure report and cleanup proof.
- `results/answer.json`: the generated answer and bounded retrieval metadata.
- `results/answer-voice.wav`: final XTTS to epoch-200 RVC output when `--voice`
  is selected.
- `logs/`: bounded, secret-redacted worker tails used for diagnosis.
- `asset-selection.json`: hashes and source selection while staging is retained.

A real voice pass requires `generation_status: generated`, an answerable/partial
same-generation verification, positive prompt evidence, HTTP 200 from the frozen
Studio `/v1/chat`, a non-empty
Speaker-model answer, an empty Ollama residency set before Voice starts, and a WAV
that is mono, finite, positive-duration, below `0.001` clipping, epoch `200`, and
RVC index rate `0.75`. The reported SHA-256 must match the response and file.

The voice is AI-generated. Do not present the WAV as a genuine statement or
recording by the target speaker.

## Timeouts and cleanup

Studio's internal RAG readiness remains bounded at 120 seconds. The outer smoke
controller allows 180 seconds for the complete Studio bootstrap; other readiness
checks retain the 120-second bound. Retrieval,
answer generation, and synthesis requests have a 660-second request bound. Real
asset hashing and model initialization can make the complete command take several
minutes before those request phases begin. The real gate intentionally performs
one direct retrieval validation and another production retrieval inside
`/v1/chat`, so a full voice run can take around ten minutes on this machine.
The voice request remains bounded to 32,768 Unicode characters: eight characters
for each of the Studio's maximum 4,096 generated tokens.

Cleanup runs in `finally`: authenticated Voice unload is attempted, each launched
root is created suspended, assigned to a Windows `KILL_ON_JOB_CLOSE` Job Object,
and only then resumed. Log pumps start after resume. Assignment, resume, or later
log-setup failure kills and reaps the child and closes pipes/sinks. All descendants
are reaped even after a dead parent, and all three ports are checked for release.
Success requires `launched_pids_alive: []`, `job_objects_closed: true`,
`ports_released: true`, `ollama_restored: true`, and
`source_integrity_verified: true` in `result.json`.

Each finalizer action is isolated. A failed Ollama restoration or follow-up
snapshot cannot skip process cleanup, source rehashing, private staging removal,
or `result.json`. The primary gate failure is preserved and cleanup diagnostics
contain only step names and exception types, never exception messages or secrets.

Process output is streamed through a bounded binary-safe matcher that holds token
prefixes across arbitrary read boundaries and EOF. Diagnostic tails are capped at
200 lines/8 KiB per entry and redacted again before being returned.

The output root is recursively replaced only when it contains the controller's
ownership marker. An unrelated existing directory is refused rather than deleted.

## Interpreting failures

- `exited before readiness`: inspect the named worker's last redacted lines in
  `result.json` and `logs/`; rebuild if a frozen dynamic import is absent.
- `readiness timed out`: distinguish active model loading from a dead child using
  the process tail and resource activity. Do not raise the per-worker bound to
  hide a reproducible startup fault.
- identity/count mismatch: rebuild the temporary RAG staging from the approved
  clean/card manifests; do not edit the Qdrant source in place.
- Ollama identity/residency failure: compare the recorded `before`, transient,
  `restoration`, and `after_cleanup` snapshots. Cleanup restores required models
  to their initial state and fails if unrelated identities changed.
- Voice validation failure: retain output with `--keep-output`, then inspect the
  reported sample rate, clipping, RVC metadata, and SHA before changing models.

Failure reports include only exit codes and the final 200 redacted lines per
process. Session secrets and corpus text are not diagnostic output.
