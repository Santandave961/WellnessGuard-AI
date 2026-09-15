"""
FastAPI backend: stress inference endpoint + companion chatbot trigger.
Companion model swappable — plug in Claude, Gemini, or a local LLM.
"""

import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the project root, if present
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    genai = None

app = FastAPI(title="WellnessGuard AI")

MODEL_PATH = os.environ.get("STRESS_MODEL_PATH", "models/stress_clf.joblib")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_model = None
_llm_client = None
_explainer = None

# Human-readable labels for the companion prompt — keep these non-clinical
# (e.g. "heart rate" not "HR_mean (bpm)") since they get surfaced to the user.
READABLE_FEATURE_NAMES = {
    "EDA_mean": "skin conductance",
    "EDA_std": "skin conductance variability",
    "HR_mean": "heart rate",
    "HR_std": "heart rate variability",
    "TEMP_mean": "skin temperature",
}


def get_explainer(model):
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(model)
    return _explainer


def get_top_drivers(model, X: pd.DataFrame, top_n: int = 2) -> list:
    """
    Returns the top_n features pushing this specific prediction toward
    "stress", as human-readable names — used to make the companion
    message specific instead of generic.
    """
    explainer = get_explainer(model)
    raw = explainer.shap_values(X)

    # shap's return shape varies by version/model type (list per class,
    # 2D array, or 3D array with a trailing class axis) — normalize to a
    # single 1D array of per-feature values for this one row.
    if isinstance(raw, list):
        row = raw[1][0] if len(raw) > 1 else raw[0][0]
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3:
            row = arr[0, :, -1]
        else:
            row = arr[0]

    contribs = list(zip(X.columns, row))
    # Only features pushing toward stress (positive SHAP value), strongest first
    positive = [c for c in contribs if c[1] > 0]
    positive.sort(key=lambda c: c[1], reverse=True)
    top = positive[:top_n] if positive else sorted(contribs, key=lambda c: abs(c[1]), reverse=True)[:top_n]

    return [READABLE_FEATURE_NAMES.get(name, name) for name, _ in top]


class SignalWindow(BaseModel):
    eda_mean: float
    eda_std: float
    hr_mean: float
    hr_std: float
    temp_mean: float
    user_id: str = "demo_user"


class StressResult(BaseModel):
    stress_prob: float
    flagged: bool
    timestamp: str
    companion_prompt: Optional[str] = None
    companion_message: Optional[str] = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def get_llm_client():
    global _llm_client
    if _llm_client is None and genai is not None and GEMINI_API_KEY:
        _llm_client = genai.Client(api_key=GEMINI_API_KEY)
    return _llm_client


def generate_companion_message(prompt: str) -> Optional[str]:
    """
    Calls the Gemini API to generate the actual check-in message.
    Returns None if no API key is configured — callers should fall back
    to a default message in that case rather than failing the request.
    """
    client = get_llm_client()
    if client is None:
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return (response.text or "").strip() or None
    except Exception as e:
        print(f"[companion] LLM call failed: {e}")
        return None


def build_companion_prompt(stress_prob: float, top_drivers: list) -> str:
    """
    Construct the check-in prompt sent to the companion LLM.
    Keep this factual/grounding — no diagnostic or clinical language,
    since this is a self-reflection tool, not a medical device.
    """
    driver_text = ", ".join(f"{name}" for name, _ in top_drivers) if top_drivers else "general signal shift"
    return (
        f"The user's physiological signals show an elevated stress pattern "
        f"(confidence {stress_prob:.0%}), driven mainly by changes in {driver_text}. "
        f"Write a brief, warm, non-clinical check-in message asking how they're doing "
        f"and offering one simple grounding suggestion (e.g. a breathing pause or a short walk). "
        f"Do not mention specific sensor values or make medical claims."
    )


@app.post("/predict", response_model=StressResult)
def predict(window: SignalWindow):
    model = get_model()
    X = pd.DataFrame([{
        "EDA_mean": window.eda_mean,
        "EDA_std": window.eda_std,
        "HR_mean": window.hr_mean,
        "HR_std": window.hr_std,
        "TEMP_mean": window.temp_mean,
    }])
    prob = float(model.predict_proba(X)[0, 1])
    flagged = prob > 0.5

    top_drivers = []
    if flagged:
        try:
            top_drivers = get_top_drivers(model, X)
        except Exception as e:
            print(f"[companion] SHAP explanation failed, using generic prompt: {e}")

    prompt = build_companion_prompt(prob, [(d, None) for d in top_drivers]) if flagged else None
    message = None
    if prompt:
        message = generate_companion_message(prompt)
        if message is None:
            # Fallback when no API key is set or the call fails —
            # keeps the dashboard functional without a live LLM.
            message = "Noticed a stress spike — want a 2-min breathing pause?"

    return StressResult(
        stress_prob=prob,
        flagged=flagged,
        timestamp=datetime.utcnow().isoformat(),
        companion_prompt=prompt,
        companion_message=message,
    )


@app.get("/health")
def health():
    return {"status": "ok"}