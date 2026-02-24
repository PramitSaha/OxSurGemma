"""
Speech-to-text module using faster-whisper (or openai-whisper as fallback).
Install: pip install faster-whisper  (preferred, has pre-built wheels)
Gradio records as WebM; we convert to WAV for Whisper if needed.
"""
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

_faster_model = None
_faster_model_size = None
_openai_model = None
_openai_model_size = None


def _webm_or_ogg_to_wav(audio_path: str) -> Optional[str]:
    """Convert WebM/OGG (e.g. from Gradio mic) to WAV for Whisper. Returns path to wav or None."""
    path = Path(audio_path)
    if not path.exists():
        return None
    suf = path.suffix.lower()
    if suf not in (".webm", ".ogg", ".opus"):
        return audio_path
    out = Path(tempfile.gettempdir()) / f"stt_{path.stem}.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(out),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return str(out) if out.exists() else None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[STT] ffmpeg convert failed: {e}", flush=True)
        return None


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    language: Optional[str] = "en",
) -> str:
    """
    Transcribe audio file to text using Whisper.

    Args:
        audio_path: Path to audio file (wav, mp3, webm, etc. — WebM from Gradio is converted to WAV)
        model_size: Whisper model size: tiny, base, small, medium, large-v3
        language: Optional language code (e.g. "en" for English)

    Returns:
        Transcribed text.
    """
    path = Path(audio_path)
    if not path.exists():
        return f"Error: audio file not found: {audio_path}"

    # Gradio mic records WebM; Whisper works better with WAV
    use_path = _webm_or_ogg_to_wav(audio_path) or audio_path

    # Prefer faster-whisper (pre-built, no compile)
    try:
        from faster_whisper import WhisperModel

        global _faster_model, _faster_model_size
        if _faster_model is None or _faster_model_size != model_size:
            print(f"[STT] Loading faster-whisper '{model_size}' ...", flush=True)
            try:
                _faster_model = WhisperModel(model_size, device="cuda", compute_type="float16")
            except Exception:
                _faster_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            _faster_model_size = model_size
            print(f"[STT] faster-whisper '{model_size}' ready.", flush=True)
        segments, _ = _faster_model.transcribe(use_path, language=language)
        # Consume generator
        text = " ".join(s.text for s in segments).strip() if segments else ""
        return text if text else "(no speech detected)"
    except ImportError:
        pass
    except Exception as e:
        print(f"[STT] faster-whisper error: {e}", flush=True)
        return f"[Voice error: {e}. Install: pip install faster-whisper]"

    # Fallback to openai-whisper
    try:
        import whisper

        global _openai_model, _openai_model_size
        if _openai_model is None or _openai_model_size != model_size:
            _openai_model = whisper.load_model(model_size)
            _openai_model_size = model_size
        result = _openai_model.transcribe(use_path, language=language, fp16=False)
        return (result.get("text") or "").strip() or "(no speech detected)"
    except ImportError:
        return "[Voice error: Install STT: pip install faster-whisper]"
    except Exception as e:
        print(f"[STT] whisper error: {e}", flush=True)
        return f"[Voice error: {e}]"

    return "[Voice error: Could not transcribe. Install: pip install faster-whisper]"
