# audio.py
from pydub import AudioSegment
import io
import soundfile as sf
import numpy as np
import librosa

def convert_to_wav(file_bytes: bytes) -> bytes:
    """Convert any audio format to WAV using pydub."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io.read()
    except Exception as e:
        raise RuntimeError(f"Failed to convert to WAV: {e}")

def read_audio(file_like_or_bytes, target_sr=16000):
    """Load, normalize, and resample audio."""
    try:
        data = io.BytesIO(file_like_or_bytes) if isinstance(file_like_or_bytes, (bytes, bytearray)) else file_like_or_bytes
        audio, sr = sf.read(data, dtype='float32')
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return np.clip(audio, -1.0, 1.0)
    except Exception as e:
        raise RuntimeError(f"Error reading audio file: {e}")