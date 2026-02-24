"""Speech-to-text and text-to-speech modules for the surgical co-pilot."""
try:
    from surgical_copilot.audio.stt import transcribe_audio
except ImportError:
    transcribe_audio = None
try:
    from surgical_copilot.audio.tts import text_to_speech
except ImportError:
    text_to_speech = None

__all__ = ["transcribe_audio", "text_to_speech"]
