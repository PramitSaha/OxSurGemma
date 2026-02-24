"""
Text-to-speech module using gTTS (Google Text-to-Speech).
Install: pip install gtts
"""
from pathlib import Path
from typing import Optional

_TTS_AVAILABLE = False


def _check_gtts():
    global _TTS_AVAILABLE
    if _TTS_AVAILABLE:
        return True
    try:
        from gtts import gTTS
        _TTS_AVAILABLE = True
        return True
    except ImportError:
        return False


def text_to_speech(
    text: str,
    output_path: str,
    lang: str = "en",
    slow: bool = False,
) -> Optional[str]:
    """
    Convert text to speech and save to file.

    Args:
        text: Text to speak
        output_path: Path to save the audio file (mp3)
        lang: Language code (default: en)
        slow: Use slower speech

    Returns:
        Path to saved audio file, or None if failed.
    """
    if not text or not text.strip():
        return None

    if not _check_gtts():
        return None

    try:
        from gtts import gTTS

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tts = gTTS(text=text.strip(), lang=lang, slow=slow)
        tts.save(str(path))
        return str(path) if path.exists() else None
    except Exception:
        return None
