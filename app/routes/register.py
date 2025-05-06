from app.services.db import get_db_connection
from app.services.inference import run_denoising, run_deepfake, get_embedding
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from app.services.audio import convert_to_wav, read_audio
from datetime import datetime
import os, uuid, io
import soundfile as sf

router = APIRouter()

SAVE_DIR = "debug_uploads"
os.makedirs(SAVE_DIR, exist_ok=True)
@router.post("/register")
async def register_user(
    audio: UploadFile = Form(...),
    user_id: str = Form(...)
):
    try:
        raw = await audio.read()
        original_filename = audio.filename or "audio_upload"
        print(f"\nReceived file: {original_filename}")

        # Step 1: Convert to WAV
        try:
            wav_bytes = convert_to_wav(raw)
            print(f"Converted to WAV successfully")
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Conversion failed: {e}"})

        # Step 2: Save WAV to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{timestamp}_{user_id}_{uuid.uuid4().hex[:6]}.wav"
        filepath = os.path.join(SAVE_DIR, unique_name)
        with open(filepath, "wb") as f:
            f.write(wav_bytes)
        print(f"File saved as: {filepath}")

        # Step 3: Print metadata
        try:
            with sf.SoundFile(filepath) as f:
                print(f" Format    : {f.format}")
                print(f"Subtype     : {f.subtype}")
                print(f"Sample rate : {f.samplerate}")
                print(f"Channels    : {f.channels}")
                print(f"Duration    : {len(f) / f.samplerate:.2f} sec")
        except Exception as e:
            print(f" Metadata read failed: {e}")

        # Step 4: Read waveform
        try:
            audio_array = read_audio(wav_bytes)
            print(f" Loaded waveform shape: {audio_array.shape}")
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Waveform load failed: {e}"})

        # Step 5: Generate attempt ID
        attempt_id = str(uuid.uuid4())

        # Step 6: Save raw input to DB
        try:
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO attempts (attempt_id, user_id, raw_audio) VALUES (?, ?, ?)",
                    (attempt_id, user_id, raw)
                )
                conn.commit()
        except Exception as e:
            print(f"Failed to save attempt: {e}")
            return JSONResponse(status_code=500, content={"error": "Failed to save attempt to database."})

        # Step 7: Deepfake detection
        # try:
        #     is_real, confidence = run_deepfake(audio_array)
        #     print(f" Deepfake detection: is_real={is_real}, confidence={confidence:.4f}")
        #     with get_db_connection() as conn:
        #         conn.execute(
        #             "INSERT INTO deepfake_results (attempt_id, is_real, confidence) VALUES (?, ?, ?)",
        #             (attempt_id, int(is_real), confidence)
        #         )
        #         conn.commit()
        # except Exception as e:
        #     print(f"Deepfake detection failed: {e}")
        #     return JSONResponse(status_code=500, content={"error": "Deepfake detection failed."})
        #
        # # Step 8: Reject if fake
        # if not is_real:
        #     return JSONResponse(
        #         status_code=400,
        #         content={"error": "Audio rejected. Detected as AI-generated."}
        #     )
        #
        # # Step 9: Return response (we stop here before denoising/embedding)
        # print("✅ Audio accepted (real human voice). Ready for denoising next.")
        # print(f" Attempt ID   : {attempt_id}")
        # print(f" Filename     : {unique_name}")
        # print(f" Duration (s) : {round(len(audio_array) / 16000, 2)}")


        # Step 10: Denoising
        try:
            denoised = run_denoising(audio_array)
            print(f" Denoising complete. Shape: {denoised.shape}")

            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO denoised_audio (attempt_id, audio) VALUES (?, ?)",
                    (attempt_id, denoised.tobytes())
                )
                conn.commit()
        except Exception as e:
            print(f"Denoising failed: {e}")
            return JSONResponse(status_code=500, content={"error": "Denoising failed."})

        # Step 8: Continue after denoising
        print("Audio denoised successfully. Ready for embedding next.")
        print(f" Attempt ID   : {attempt_id}")
        print(f" Filename     : {unique_name}")
        print(f" Duration (s) : {round(len(audio_array) / 16000, 2)}")

        # Step: Extract embedding from denoised audio
        emb = get_embedding(denoised)
        print(f"Embedding extracted. Shape: {emb.shape}")

        # Step: Store embedding in DB
        try:
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO embeddings (attempt_id, user_id, embedding) VALUES (?, ?, ?)",
                    (attempt_id, user_id, emb.tobytes())
                )
                conn.commit()
            print("Embedding saved to database.")
        except Exception as e:
            print(f"Failed to save embedding: {e}")
            return JSONResponse(status_code=500, content={"error": "Failed to save embedding to DB."})

    except Exception as e:
        print(f" Unhandled error in /register: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
