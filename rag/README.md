# Transcript grounded answers

The current evidence path retrieves clean transcript spans, verifies claims with
`qwen3:4b`, then returns an answerability decision. In strict mode, unsupported or
invalid evidence cannot authorize generation. Optional cards only guide searches.
Fallback and free generation are separately selected application modes.

## Prepare your index

Supply permitted, reviewed segments in `data/clean/` and metadata in `data/meta/`.
Configure the collection and model in `config.yaml`, then run:

```powershell
uv run python -m rag.index
```

This rebuilds the configured local Qdrant collection. Back up an existing index
before rebuilding. The index and corpus are not included in the source release.

## Runtime

The application uses `rag.evidence_pipeline`, `rag.verifier`, and the local retrieval
worker. `GET /retrieve?q={question}&mode=clean|clean_with_card_hints` returns the
structured result. Model files and the source corpus must be configured separately.
Low-level `ask*` utilities include earlier research approaches; use the application's
strict mode when testing the current evidence gate.

## Evaluation

The `eval.rag_reliability` and `eval.card_eval` tools require a separately supplied
evaluation suite and its matching clean sources. Private suite JSON files are ignored
by Git. Source-only tests do not substitute for live-model and corpus evaluation.
See [validation scope](../docs/VALIDATION.md).
