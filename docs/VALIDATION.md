# Validation scope

This is a generic source snapshot, not a pretrained model or production installer release.

## Local verification

Verification date: 2026-09-13. Host: Windows, Python 3.12 and .NET SDK 9.0.301.

- Python: 753 passed, 3 skipped, 5 slow tests deselected.
- .NET: 400 passed, no skips or failures in the final run.
- `uv lock --check --offline`: passed, 173 packages resolved.
- Publication scan: 479 tracked files, approximately 4.9 MB; 14 notebooks have
  empty saved outputs. No original subject identifiers or recognized secret-token
  patterns were found. Pattern scanning is not a guarantee against every secret format.

```powershell
python -m pytest tests -m "not slow" -q
dotnet test app/YouTuberStudio.sln -c Release
```

The Python run uses an existing development environment. It does not prove that a
fresh installation of every GPU dependency succeeds on another machine. The locked
dependency manifest is checked separately. Windows tests use generated Release
worker fixtures, not real pretrained models.

## Excluded private evidence

The original `eval/fixtures/verified_card_eval_v1.json` and
`eval/fixtures/rag_reliability_v1.json` contain real-source excerpts and are not
distributed. Ten test modules that depend on those suites are not collected until
both suites are supplied locally. Their implementation and strict assertions remain
in the source for private-corpus evaluation. Other evaluation and runtime tests run
without those files. Three packaged-worker tests can skip when their executables
are absent. Five slow tests are deselected by the command above.

The small prompt examples in `training/eval_prompts.json` and `eval/probe_topics.json`
are synthetic examples, not reference answers or benchmark results. Known source
video identifiers in examples are replaced with synthetic identifiers. Never treat
those identifiers as real evidence or published evaluation results.

## Compatibility

Public namespaces use `YouTuber` / `youtuber`; model roles use `Speaker` / `speaker`.
Environment variable names use `YOUTUBER_`. These names and the generic prompt
templates differ from the private research configuration. Regenerate your model
manifests and use matching training/inference templates rather than assuming that
older artifacts can be substituted unchanged.

Colab runners keep Git checkpoints off by default. The explicit `--checkpoint`
option is intended only for a separately configured private data repository.
The source repository ignores generated data, local evidence suites, recordings,
weights, secrets, logs, and build outputs.

No live retraining, real voice similarity evaluation, full-corpus accuracy run,
or clean-machine production installer acceptance is claimed by these source tests.

## Türkçe

Bu doğrulama kaynak kodu ve yapay test verileriyle sınırlıdır. Gerçek alıntılar
içeren iki özel değerlendirme dosyası yayımlanmamıştır; bunlara bağlı on test modülü
dosyalar sağlanana kadar toplanmaz. Eski model ve ayarlar, yeni genel adlara ve
istemlere uyarlanmadan kullanılmamalıdır. Yazılım testlerinin geçmesi gerçek ses
benzerliği, model doğruluğu veya bütün bilgisayarlarda kurulum başarısı anlamına gelmez.
