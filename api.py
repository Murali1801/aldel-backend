"""
ALDEL Behavioral Biometrics Backend
report.aldel.org student/parent login - mouse, keystroke, form interaction monitoring
"""

import json
import os
from datetime import datetime
from urllib.request import Request, urlopen
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE.parent / "backend" / "pretrained_biometrics.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE / "pretrained_biometrics.pkl"

app = FastAPI(title="ALDEL Behavioral API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model_data = None
ALDEL_ADMIN = {"admin": "aldel_admin_2024"}
aldel_sessions = []
aldel_attempts = []


@app.on_event("startup")
async def load_model():
    global model_data
    if MODEL_PATH.exists():
        m = joblib.load(MODEL_PATH)
        model_data = m if isinstance(m, dict) else {"model": m}
    else:
        model_data = {"model": None}


def features_to_array(avg_dwell, avg_flight, std_dwell, std_flight, mouse_speed):
    std_d = std_dwell if std_dwell and std_dwell > 0 else 12
    std_f = std_flight if std_flight and std_flight > 0 else 18
    return np.array([[float(avg_dwell), float(avg_flight), std_d, std_f, float(mouse_speed)]])


def raw_to_risk(raw):
    r = np.clip(float(raw), -0.5, 0.5)
    return int(100 * (1 - (r + 0.5)))


class AldelVerifyPayload(BaseModel):
    page: str = "student"
    avg_dwell: float = Field(125, ge=0, le=2000)
    avg_flight: float = Field(150, ge=0, le=5000)
    std_dwell: float = Field(12, ge=0, le=200)
    std_flight: float = Field(18, ge=0, le=300)
    mouse_speed: float = Field(400, ge=0, le=2000)
    mouse_path: list = []
    clicks: list = []
    keystrokes: int = 0


class AldelAdminLogin(BaseModel):
    username: str
    password: str


@app.websocket("/ws/aldel")
async def aldel_websocket(ws: WebSocket):
    await ws.accept()
    session = {"id": len(aldel_sessions) + 1, "started": datetime.utcnow().isoformat(), "events": [], "page": None}
    aldel_sessions.append(session)
    try:
        while True:
            data = await ws.receive_json()
            session["events"].append(data)
            if "page" in data:
                session["page"] = data.get("page")
            if len(session["events"]) > 5000:
                session["events"] = session["events"][-4000:]
    except WebSocketDisconnect:
        pass


@app.post("/aldel/verify")
async def aldel_verify(p: AldelVerifyPayload):
    if not model_data or model_data["model"] is None:
        raise HTTPException(503, "Model not loaded")
    model = model_data["model"]
    X = features_to_array(p.avg_dwell, p.avg_flight, p.std_dwell, p.std_flight, p.mouse_speed)
    pred = model.predict(X)[0]
    raw = model.decision_function(X)[0]
    risk = raw_to_risk(raw)
    granted = pred == 1
    rec = {
        "id": len(aldel_attempts) + 1,
        "timestamp": datetime.utcnow().isoformat(),
        "page": p.page,
        "risk_score": risk,
        "raw_score": float(raw),
        "access_granted": granted,
        "biometrics": {"avg_dwell": p.avg_dwell, "avg_flight": p.avg_flight, "std_dwell": p.std_dwell, "std_flight": p.std_flight, "mouse_speed": p.mouse_speed},
    }
    aldel_attempts.insert(0, rec)
    aldel_attempts[:] = aldel_attempts[:200]

    admin_url = os.environ.get("ALDEL_ADMIN_URL")
    if admin_url:
        try:
            req = Request(
                f"{admin_url.rstrip('/')}/api/log",
                data=json.dumps(rec).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urlopen(req, timeout=2)
        except Exception:
            pass

    return {"access_granted": granted, "risk_score": risk, "status": "Granted" if granted else "Impersonation Detected - Access Restricted"}


@app.get("/aldel/attempts")
async def get_aldel_attempts():
    return {"attempts": aldel_attempts, "sessions": len(aldel_sessions)}


@app.post("/aldel/admin/login")
async def admin_login(p: AldelAdminLogin):
    if ALDEL_ADMIN.get(p.username) != p.password:
        raise HTTPException(401, "Invalid credentials")
    return {"token": "aldel_admin_ok", "username": p.username}


@app.get("/aldel/health")
async def health():
    return {"status": "ok", "model_loaded": model_data and model_data.get("model") is not None}
