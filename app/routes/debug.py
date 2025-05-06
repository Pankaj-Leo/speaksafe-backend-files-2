from fastapi import APIRouter, UploadFile, Form
import io
from app.services.audio import read_audio

router = APIRouter()

@router.post("/debug-audio")
async def debug_audio(audio: UploadFile = Form(...)):
    raw = await audio.read()
    try:
        print(f"📦 Received audio: {audio.filename}, type: {audio.content_type}, size: {len(raw)} bytes")
        signal = read_audio(io.BytesIO(raw))
        print(f"✅ Decoded audio: {signal.shape}, dtype: {signal.dtype}")
        return {"status": "success", "length": len(signal)}
    except Exception as e:
        print(f"❌ Failed to decode: {e}")
        return {"status": "fail", "error": str(e)}