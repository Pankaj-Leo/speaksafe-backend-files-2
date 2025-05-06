import os
import numpy as np
import joblib
import librosa
import tensorflow as tf
from keras import Input, Model
from keras.src.layers import Conv2D, Dense
from keras.src.saving import load_model
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
# ==================== CONFIG ====================
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
FRAME_LEN, FRAME_STEP = 0.025, 0.01
NUM_MFCC, MAX_FRAMES = 128, 300
INPUT_SHAPE = (134, MAX_FRAMES, 1)
CHUNK = SAMPLE_RATE * 2  # 2 seconds

# ==================== CUSTOM LOSS ====================
def spectral_convergence(y_true, y_pred):
    stft_t = tf.signal.stft(tf.squeeze(y_true, -1), 256, 128)
    stft_p = tf.signal.stft(tf.squeeze(y_pred, -1), 256, 128)
    return tf.norm(tf.abs(stft_t) - tf.abs(stft_p)) / (tf.norm(tf.abs(stft_t)) + 1e-9)

def combined_loss(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    spec = spectral_convergence(y_true, y_pred)
    return mae + 0.5 * spec

# ==================== LOAD MODELS ====================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

deepfake_model = joblib.load(os.path.join(MODEL_DIR,"best_random_forest_model.joblib"))

denoise_model = load_model(os.path.join(MODEL_DIR, "wavenet_model_5.keras"),
                          custom_objects={"combined_loss": combined_loss}, compile=False)
# denoise_model = load_model(
#     os.path.join(MODEL_DIR, "waveunet_model_5.h5"),
#     custom_objects={"combined_loss": combined_loss, "spectral_convergence": spectral_convergence},
#     compile=False
# )

# ==================== DENOISING ====================
def run_denoising(audio: np.ndarray) -> np.ndarray:
    """
    Run denoising on a waveform using Wave-U-Net.
    Args:
        audio (np.ndarray): 1D waveform
    Returns:
        np.ndarray: Denoised waveform
    """
    if len(audio) > CHUNK:
        audio = audio[:CHUNK]
    elif len(audio) < CHUNK:
        audio = np.pad(audio, (0, CHUNK - len(audio)))

    audio = np.clip(audio, -1.0, 1.0)
    input_tensor = audio.astype(np.float32)[None, ..., None]
    denoised = denoise_model.predict(input_tensor)[0, :, 0]
    return denoised

# ==================== FEATURE EXTRACTION ====================
def get_features(audio: np.ndarray, sr=16000) -> pd.DataFrame:
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))
    rms = np.mean(librosa.feature.rms(y=audio))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)

    features = {
        'chroma_stft': chroma_stft,
        'rms': rms,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'rolloff': rolloff,
        'zero_crossing_rate': zero_crossing_rate,
    }
    for i in range(20):
        features[f'mfcc{i+1}'] = mfcc[i]

    return pd.DataFrame([features])

def get_combined_features(audio: np.ndarray, sr=16000, n_mfcc=128, max_frames=300) -> np.ndarray:
    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc[:, :max_frames]
    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])))

    # Additional spectral features (1D → replicate across time axis)
    extras = [
        np.mean(librosa.feature.chroma_stft(y=audio, sr=sr)),
        np.mean(librosa.feature.rms(y=audio)),
        np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
        np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)),
        np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
        np.mean(librosa.feature.zero_crossing_rate(y=audio))
    ]
    extras = np.array(extras).reshape(-1, 1)
    extras = np.repeat(extras, max_frames, axis=1)

    # Combine: (n_mfcc + 6, max_frames)
    combined = np.vstack([mfcc, extras])
    return combined.T[..., np.newaxis]  # Final shape: (max_frames, total_features, 1)

# ==================== EMBEDDING MODEL ====================
def build_embedding_model():
    inp = Input(INPUT_SHAPE)
    x = Conv2D(3, (3, 3), padding="same", activation="relu")(inp)
    base = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=x, pooling="avg")
    emb = Dense(256, activation=None)(base.output)
    return Model(inp, emb)

embed_model = build_embedding_model()

# ==================== INFERENCE ====================
def run_deepfake(audio: np.ndarray, sr=16000) -> (bool, float):
    try:
        features_df = get_features(audio, sr=sr)
        prob = deepfake_model.predict_proba(features_df)[0]
        is_real = int(np.argmax(prob)) == 0
        confidence = float(prob[0])
        return is_real, confidence
    except Exception as e:
        print(f"Deepfake detection failed: {e}")
        raise RuntimeError(f"Deepfake detection failed: {e}")

def split_into_chunks(audio: np.ndarray, num_chunks: int = 5) -> list:
    length = len(audio)
    chunk_size = length // num_chunks
    return [audio[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]


def get_embedding(audio: np.ndarray, segment: bool = True) -> np.ndarray:
    if segment:
        # 5-segment logic for registration
        chunks = split_into_chunks(audio, num_chunks=5)
        features = [get_combined_features(chunk, sr=SAMPLE_RATE) for chunk in chunks]
        features = np.array(features)  # Shape: (5, 300, 134, 1)
        features = np.transpose(features, (0, 2, 1, 3))  # Shape: (5, 134, 300, 1)
        embeddings = embed_model.predict(features)  # Output: (5, 256)
        return np.mean(embeddings, axis=0)

    else:
        # Single full-clip logic for /verify
        feature = get_combined_features(audio, sr=SAMPLE_RATE)  # (300, 134, 1)
        feature = np.transpose(feature, (1, 0, 2))  # (134, 300, 1)
        embedding = embed_model.predict(feature[None])  # Output: (1, 256)
        return embedding[0]