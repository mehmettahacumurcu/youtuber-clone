# RVC Colab Proof of Concept

This experiment keeps the current XTTS model unchanged. XTTS produces the Turkish delivery; the trained RVC model attempts to correct only the voice/timbre.

Use the generated files:

- Colab notebook: `training/rvc_finetune_colab.ipynb`
- Upload bundle: `data/rvc_poc/speaker_rvc_poc_24m.zip`

The bundle is private target-speaker audio. Keep it in your own Drive and do not publish generated speech as a genuine statement by the real speaker.

## 1. Upload the bundle

Create this Google Drive folder:

```text
MyDrive/speaker_rvc/
```

Upload the ZIP without renaming it:

```text
MyDrive/speaker_rvc/speaker_rvc_poc_24m.zip
```

## 2. Start Colab

1. Upload or open `training/rvc_finetune_colab.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > GPU**.
3. Run cells from top to bottom.
4. Leave this setting unchanged:

```python
RUN_FULL_DATASET = False
```

The notebook checks the GPU, pins Applio to commit `3e5e248c2bd003f65f3e128dd12677533def27be`, verifies every bundle hash, and links the model log directory to Drive before training starts.

## 3. Training and recovery

The POC defaults are:

```text
RVC v2 / 32 kHz / RMVPE / ContentVec / HiFi-GAN
200 epochs / batch 8 / checkpoint every 25 epochs
```

If CUDA runs out of memory:

1. Keep the same model name, sample rate, embedder, vocoder, and dataset.
2. Change `BATCH_SIZE = 8` to `6`.
3. Rerun from the training cell.
4. If batch 6 also fails, use batch 4.

Do not lower the sample rate or silently switch models to solve an OOM.

If Colab disconnects, remount Drive and rerun the cells from the top. Completed preprocessing and extraction stages have Drive markers, and training resumes from the model's existing Drive log directory. Do not change `MODEL_NAME` when resuming.

## 4. Choose a checkpoint

After training, the notebook voices probe 1 with epochs:

```text
100 / 150 / 200
```

Listen to the raw XTTS WAV first, then the three converted WAVs. Pick the earliest checkpoint whose identity is better without metallic noise, broken consonants, pitch jumps, or rhythm damage.

Set the selected value, for example:

```python
CHOSEN_EPOCH = 150
```

Rerun the checkpoint-choice and index-sweep cells.

If an expected checkpoint is absent, the notebook stops and prints the discovered epochs. Do not substitute a different epoch silently.

## 5. Compare index rates

The chosen checkpoint converts all five fixed XTTS probes at:

```text
0.25 — protects more of the original XTTS clarity
0.50 — balanced starting point
0.75 — pushes harder toward target timbre
```

For every probe, record:

- identity: worse / same / better;
- intelligibility: broken / acceptable / clean;
- artifacts: none / mild / unacceptable;
- preferred index rate.

The POC passes only when you prefer the target identity on at least four of five probes and the automated duration, clipping, Turkish CER, and ECAPA checks do not show a regression.

## 6. Return the results

The notebook writes everything under:

```text
MyDrive/speaker_rvc/speaker_rvc_poc_v1/
```

Return this file for review:

```text
MyDrive/speaker_rvc/speaker_rvc_poc_v1/speaker_rvc_poc_v1_results.zip
```

It contains the `.pth`, `.index`, fixed raw probes, every converted comparison, training/bundle manifests, automatic metrics, and the listening sheet.

If Colab's isolated Whisper/ECAPA evaluation cannot install, the training artifacts and basic audio metrics are still preserved. After extracting the returned ZIP locally, the fallback is:

```powershell
& ".venv\Scripts\python.exe" -m training.rvc_eval `
  --bundle-root "<extracted bundle directory>" `
  --conversion-manifest "<comparison directory>\conversion_manifest.json" `
  --output "<comparison directory>\metrics_full.json" `
  --allow-model-failure
```

## 7. Do not start the full run yet

Keep `RUN_FULL_DATASET = False` until the fixed five-probe report has been reviewed. The next branch is chosen from evidence:

- pass: prepare the full 74.14-minute run;
- identity gain with artifacts: earlier checkpoint or lower index rate;
- clean but weak identity: expand to 40–50 minutes;
- no useful identity gain: stop RVC and move to restricted XTTS decoder adaptation.
