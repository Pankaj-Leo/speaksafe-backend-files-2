from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import register, verify, debug
from app.services.db import init_db

app = FastAPI(title="SpeakSafe Voice Backend")

# CORS: Allow requests from your frontend (e.g., Next.js on port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Route registration
app.include_router(register.router)
app.include_router(verify.router)


app.include_router(debug.router)

#  Initialize DB tables if they don't exist
init_db()