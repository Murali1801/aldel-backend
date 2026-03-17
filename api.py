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
    """
    Map Isolation Forest decision_function to risk 0-100.
    Genuine (inlier, raw > 0) → low risk 0-35.
    Anomaly (outlier, raw < 0) → high risk 35-100.
    """
    r = np.clip(float(raw), -0.5, 0.5)
    if r >= 0:
        return int(70 * (0.5 - r))  # raw 0.5→0, raw 0→35
    return int(35 + 65 * (0 - r) / 0.5)  # raw 0→35, raw -0.5→100


class AldelVerifyPayload(BaseModel):
    page: str = "student"
    avg_dwell: float = Field(125, ge=0, le=10000)
    avg_flight: float = Field(150, ge=0, le=30000)
    std_dwell: float = Field(12, ge=0, le=1000)
    std_flight: float = Field(18, ge=0, le=1000)
    mouse_speed: float = Field(400, ge=0, le=5000)
    key_events: list = []
    mouse_path: list = []
    clicks: list = []
    keystrokes: int = 0
    duration_ms: int = 0


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


def bot_heuristics(p, risk: int) -> bool:
    """
    Genuine: move to username -> click -> type -> move to password -> click -> type
    -> move to captcha -> click -> type -> move to login -> click. Takes 5-15+ sec.
    High dwell (e.g. 2000ms) = strong genuine signal (humans pause; bots paste with low dwell).
    Bot: paste/fast fill, captcha API. Under 4 sec, few key_events.
    """
    avg_dwell = getattr(p, "avg_dwell", 125) or 125
    if avg_dwell >= 1500:
        return False  # High dwell = genuine human
    mp = getattr(p, "mouse_path", []) or []
    cl = getattr(p, "clicks", []) or []
    ke = getattr(p, "key_events", []) or []
    duration = getattr(p, "duration_ms", 0) or 0
    mp_len, cl_len, ke_len = len(mp), len(cl), len(ke)
    if mp_len < 5 and ke_len > 8:
        return True
    if cl_len == 0 and ke_len > 10 and mp_len < 25:
        return True
    if ke_len >= 4:
        dwells = [e.get("up", 0) - e.get("down", 0) for e in ke if isinstance(e, dict) and "up" in e and "down" in e]
        if dwells and all(0 <= d <= 20 for d in dwells) and len(set(round(d) for d in dwells)) <= 2:
            return True
    if duration > 0 and duration < 4000 and (ke_len > 5 or cl_len >= 1):
        return True
    if risk >= 50 and (mp_len < 40 or cl_len < 3):
        return True
    if avg_dwell <= 65 and getattr(p, "std_dwell", 12) <= 8 and ke_len > 5:
        return True
    return False


@app.post("/aldel/verify")
async def aldel_verify(p: AldelVerifyPayload):
    if not model_data or model_data["model"] is None:
        raise HTTPException(503, "Model not loaded")
    model = model_data["model"]
    X = features_to_array(p.avg_dwell, p.avg_flight, p.std_dwell, p.std_flight, p.mouse_speed)
    pred = model.predict(X)[0]
    raw = model.decision_function(X)[0]
    risk = raw_to_risk(raw)
    bot_like = bot_heuristics(p, risk)
    high_dwell = (getattr(p, "avg_dwell", 0) or 0) >= 1500  # Strong genuine signal
    risk_ok = risk <= (65 if high_dwell else 45)
    granted = (pred == 1 or risk_ok) and not bot_like
    rec = {
        "id": len(aldel_attempts) + 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "page": p.page,
        "risk_score": risk,
        "raw_score": float(raw),
        "access_granted": granted,
        "bot_heuristic_blocked": bot_like,
        "biometrics": {
            "avg_dwell": p.avg_dwell,
            "avg_flight": p.avg_flight,
            "std_dwell": p.std_dwell,
            "std_flight": p.std_flight,
            "mouse_speed": p.mouse_speed,
            "duration_ms": getattr(p, "duration_ms", 0),
            "mouse_path_len": len(getattr(p, "mouse_path", []) or []),
            "clicks_len": len(getattr(p, "clicks", []) or []),
            "key_events_len": len(getattr(p, "key_events", []) or []),
        },
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
