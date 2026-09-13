import json

import numpy as np
import soundfile as sf

from pipeline.clips import export_clips, segment_id


def test_segment_id_format():
    assert segment_id(7, 1.5, 3.25) == "0007_1.50-3.25"


def _make_wav(path, seconds, sr=16000):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False, dtype=np.float32)
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, subtype="PCM_16")


def test_export_clips_only_keep_and_borderline(settings):
    vid = "vid1"
    _make_wav(settings.paths.audio_dir / f"{vid}.wav", seconds=10.0)
    idn = {
        "video_id": vid,
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "S0", "similarity": 0.8, "decision": "keep"},
            {"start": 2.0, "end": 4.0, "speaker": "S1", "similarity": 0.2, "decision": "drop"},
            {"start": 4.0, "end": 6.0, "speaker": "S0", "similarity": 0.6, "decision": "borderline"},
        ],
    }
    (settings.paths.identify_dir / f"{vid}.json").write_text(json.dumps(idn), encoding="utf-8")

    written = export_clips(vid, settings)

    assert len(written) == 2  # keep + borderline, not drop
    keep_clip = settings.paths.data_dir / "clips" / vid / "0000_0.00-2.00.wav"
    border_clip = settings.paths.data_dir / "clips" / vid / "0002_4.00-6.00.wav"
    assert keep_clip.exists() and border_clip.exists()
    info = sf.info(str(keep_clip))
    assert abs(info.frames / info.samplerate - 2.0) < 0.05
