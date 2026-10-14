"""Speaker diarization (optional dependencies: pip install -r requirements-diarize.txt).

Two backends, chosen with DIARIZER=fast|pyannote (default: fast when sherpa-onnx is installed, else pyannote):
  fast     sherpa-onnx (pyannote segmentation + WeSpeaker embeddings as ONNX) - ~10x faster on CPU, models auto-download to models/
  pyannote pyannote/speaker-diarization-3.1 - most accurate, slow on CPU; needs HF_TOKEN and the model terms accepted on huggingface.co
Returns [(start, end, speaker_id)] in wall-clock seconds of the source file, cached as JSON.
"""
import json, os, subprocess, tarfile, tempfile, time, wave
from pathlib import Path
import requests
from env import load_env

load_env()
MODEL = "pyannote/speaker-diarization-3.1"
MODELS = Path(os.environ.get("DATA_DIR") or Path(__file__).parent) / "models"  # a Docker volume when DATA_DIR=/data
SEG_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_FILE = "wespeaker_en_voxceleb_resnet34_LM.onnx"  # English speaker embeddings, 26 MB
EMB_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/" + EMB_FILE


_pipe = None


def load_pipeline():
    """Load pyannote once per process (a batch of episodes reuses it) and put it on a GPU when there is one.
    DIARIZE_DEVICE=cpu|cuda|mps overrides auto-detection. A GPU makes this step ~30-50x faster than CPU."""
    global _pipe
    if _pipe: return _pipe
    token = os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN missing in .env (needed for pyannote)")
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise RuntimeError("pyannote not installed: pip install -r requirements-diarize.txt")
    try:
        try:
            pipe = Pipeline.from_pretrained(MODEL, token=token)
        except TypeError:  # older pyannote
            pipe = Pipeline.from_pretrained(MODEL, use_auth_token=token)
    except Exception as e:
        if "403" in str(e) or "gated" in str(e).lower():
            raise RuntimeError("HuggingFace denied access. Accept the terms on pyannote/speaker-diarization-3.1, "
                               "pyannote/segmentation-3.0 and pyannote/speaker-diarization-community-1, and give the token "
                               "'Read access to public gated repos' (or use a Read-type token).") from e
        raise
    import torch
    dev = os.environ.get("DIARIZE_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    pipe.to(torch.device(dev))
    torch.set_num_threads(os.cpu_count() or 4)  # torch defaults to fewer threads than the machine has
    for attr in ("segmentation_batch_size", "embedding_batch_size"):  # bigger batches = fewer slow calls
        if hasattr(pipe, attr): setattr(pipe, attr, 32)
    print(f"  diarization device: {dev}", flush=True)
    _pipe = pipe
    return pipe


def _fetch(url, dst):
    print(f"  downloading {Path(dst).name}...", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = Path(str(dst) + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20): f.write(chunk)
        tmp.replace(dst)


def _sherpa_models():
    """Download the two small ONNX models once into models/ and return (segmentation, embedding) paths."""
    MODELS.mkdir(exist_ok=True)
    seg = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    if not seg.exists():
        tar = MODELS / "segmentation.tar.bz2"
        _fetch(SEG_URL, tar)
        with tarfile.open(tar) as t: t.extractall(MODELS)
        tar.unlink()
    emb = MODELS / EMB_FILE
    if not emb.exists(): _fetch(EMB_URL, emb)
    return seg, emb


def _threads():
    """DIARIZE_THREADS caps CPU use (set it low on a shared server); default = all cores up to 8."""
    return int(os.environ.get("DIARIZE_THREADS") or min(os.cpu_count() or 4, 8))


def _diarize_sherpa(wav_path, num_speakers, log=print):
    import numpy as np, sherpa_onnx
    seg, emb = _sherpa_models()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg), window_shift_ratio=float(os.environ.get("DIARIZE_SHIFT", 0.5))),  # 0.1 = most accurate, ~5x slower
            num_threads=_threads()),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=_threads()),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=0.5),
        min_duration_on=0.3, min_duration_off=0.5)
    if not config.validate(): raise RuntimeError("sherpa-onnx diarization config is invalid (check models/)")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    with wave.open(str(wav_path)) as w:
        assert w.getframerate() == sd.sample_rate and w.getnchannels() == 1
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    last = [-1]

    def progress(done, total):
        pct = int(100 * done / max(total, 1)) // 10 * 10  # log every 10%
        if pct != last[0]: last[0] = pct; log(f"  diarization {pct}%")
        return 0  # 0 = keep going
    return [(r.start, r.end, f"SPEAKER_{r.speaker:02d}") for r in sd.process(samples, callback=progress).sort_by_start_time()]


def _backend():
    b = os.environ.get("DIARIZER")
    if b: return b
    try:
        import sherpa_onnx  # noqa: F401
        return "fast"
    except ImportError:
        return "pyannote"


def diarize(src, start, end, cache_dir, num_speakers=None, log=print):
    backend = _backend()
    cache = Path(cache_dir) / f"diarization-{start:.0f}-{end:.0f}-{num_speakers or 'auto'}{'-fast' if backend == 'fast' else ''}.json"
    if cache.exists():
        return [tuple(t) for t in json.loads(cache.read_text())]
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"  # 16 kHz mono wav so the backends need no mp3 decoder
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(end - start), "-i", str(src),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
        t0 = time.time()
        if backend == "fast":
            turns = [(start + a, start + b, spk) for a, b, spk in _diarize_sherpa(wav, num_speakers, log)]
        else:
            kw = {"num_speakers": num_speakers} if num_speakers else {}
            res = load_pipeline()(str(wav), **kw)
            ann = getattr(res, "speaker_diarization", res)  # pyannote 4 wraps the annotation
            turns = [(start + seg.start, start + seg.end, spk) for seg, _, spk in ann.itertracks(yield_label=True)]
        log(f"  diarization ({backend}) took {time.time() - t0:.0f}s for {end - start:.0f}s audio")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(turns))
    return turns
