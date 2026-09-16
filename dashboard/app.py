"""
WellnessGuard AI — Streamlit dashboard
Stress timeline + driver breakdown + companion conversation log.
Deploy target: Streamlit Cloud (no runtime.txt, unpinned requirements —
per your usual deployment pattern).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # override=True so edits to .env take effect without a full restart
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    genai = None

st.set_page_config(page_title="WellnessGuard AI", layout="wide")
st.title("🧠 WellnessGuard AI — Stress Pattern Detector & Companion")

st.markdown(
    "Detects stress episodes from wearable signals (EDA, HR, temperature) "
    "and logs companion check-ins. Demo mode uses synthetic data — swap in "
    "WESAD/SWELL-KW for real evaluation."
)

# Persisted so it survives st.rerun() — a one-shot st.sidebar.error() call
# gets wiped the instant rerun() fires, before it's ever visible.
if "gemini_error" not in st.session_state:
    st.session_state.gemini_error = None

# --- Gemini setup ---
# Checks local .env first (dev), then Streamlit Cloud's secrets manager (prod).
default_key = os.environ.get("GEMINI_API_KEY", "")
if not default_key:
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass  # no secrets.toml configured — fine locally

with st.sidebar:
    st.subheader("Companion settings")
    if default_key:
        st.success("Gemini key loaded ✓")
        gemini_key = default_key
    else:
        gemini_key = st.text_input("Gemini API key", type="password")
        st.caption("Get a key at aistudio.google.com/apikey, or add it to .env / Streamlit secrets to skip this.")

    if st.session_state.gemini_error:
        st.error(st.session_state.gemini_error)


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key: str):
    if genai is None or not api_key:
        return None
    return genai.Client(api_key=api_key)


def generate_companion_message(stress_prob: float, top_drivers: list = None) -> str:
    driver_text = ", ".join(top_drivers) if top_drivers else "a shift in your signals"
    prompt = (
        f"The user's physiological signals show an elevated stress pattern "
        f"(confidence {stress_prob:.0%}), driven mainly by changes in {driver_text}. "
        f"Write a brief, warm, non-clinical check-in message (1-2 sentences) asking how "
        f"they're doing and offering one simple grounding suggestion (e.g. a breathing "
        f"pause or a short walk). Do not mention specific sensor values or make medical claims."
    )

    client = get_gemini_client(gemini_key)
    if client is None:
        return "Noticed a stress spike — want a 2-min breathing pause?"

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = (response.text or "").strip()
        if text:
            st.session_state.gemini_error = None
            return text
        # Empty response with no exception — Gemini's safety filters can do
        # this silently. Surface it so it's distinguishable from a real crash.
        st.session_state.gemini_error = (
            "Gemini returned an empty response (possibly filtered) — showing fallback message."
        )
        return "Noticed a stress spike — want a 2-min breathing pause?"
    except Exception as e:
        st.session_state.gemini_error = f"Gemini call failed: {type(e).__name__}: {e}"
        return "Noticed a stress spike — want a 2-min breathing pause?"

# --- Demo data (replace with real inference pipeline output) ---
@st.cache_data
def load_demo_data():
    n = 500
    idx = pd.date_range(end=datetime.now(), periods=n, freq="1min")
    rng = np.random.default_rng(42)
    hr = 70 + rng.normal(0, 5, n).cumsum() * 0.05
    eda = 2 + rng.normal(0, 0.3, n).cumsum() * 0.02
    stress_prob = 1 / (1 + np.exp(-(0.05 * (hr - 75) + 0.3 * (eda - 2))))
    return pd.DataFrame({
        "timestamp": idx, "HR": hr, "EDA": eda, "stress_prob": stress_prob,
        "flagged": stress_prob > 0.6,
    })

df = load_demo_data()

col1, col2, col3 = st.columns(3)
col1.metric("Current Stress Score", f"{df['stress_prob'].iloc[-1]:.0%}")
col2.metric("Episodes Today", int(df["flagged"].sum()))
col3.metric("Avg HR (bpm)", f"{df['HR'].mean():.0f}")

st.subheader("Stress Timeline")
fig = px.line(df, x="timestamp", y="stress_prob", title="Stress probability over time")
flagged_points = df[df["flagged"]]
fig.add_scatter(x=flagged_points["timestamp"], y=flagged_points["stress_prob"],
                 mode="markers", marker=dict(color="red", size=8), name="Flagged episode")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Signal Breakdown")
c1, c2 = st.columns(2)
c1.plotly_chart(px.line(df, x="timestamp", y="HR", title="Heart Rate"), use_container_width=True)
c2.plotly_chart(px.line(df, x="timestamp", y="EDA", title="Electrodermal Activity"), use_container_width=True)

st.subheader("💬 Companion Check-in Log")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
    st.session_state.last_checkin_flagged = False

# Auto-generate a check-in the moment the latest reading is flagged
latest_flagged = bool(df["flagged"].iloc[-1])
if latest_flagged and not st.session_state.last_checkin_flagged:
    latest_prob = float(df["stress_prob"].iloc[-1])
    msg = generate_companion_message(latest_prob)
    st.session_state.chat_log.append({"time": datetime.now().strftime("%H:%M"), "msg": msg})
st.session_state.last_checkin_flagged = latest_flagged

if st.button("🔔 Force a check-in now (demo)"):
    msg = generate_companion_message(float(df["stress_prob"].iloc[-1]))
    st.session_state.chat_log.append({"time": datetime.now().strftime("%H:%M"), "msg": msg})
    st.rerun()

for entry in st.session_state.chat_log:
    st.chat_message("assistant").write(f"**{entry['time']}** — {entry['msg']}")

user_input = st.chat_input("Reply to your companion...")
if user_input:
    st.session_state.chat_log.append({"time": datetime.now().strftime("%H:%M"), "msg": f"You: {user_input}"})

    client = get_gemini_client(gemini_key)
    if client is not None:
        try:
            reply_prompt = (
                f"You are a warm, non-clinical wellness companion. The user just replied: "
                f"\"{user_input}\". Respond briefly (1-2 sentences), supportively, no medical claims."
            )
            reply = client.models.generate_content(model="gemini-2.5-flash", contents=reply_prompt)
            reply_text = (reply.text or "").strip()
            if reply_text:
                st.session_state.chat_log.append({"time": datetime.now().strftime("%H:%M"), "msg": reply_text})
                st.session_state.gemini_error = None
            else:
                st.session_state.gemini_error = (
                    "Gemini returned an empty reply (possibly filtered) — no message added."
                )
        except Exception as e:
            st.session_state.gemini_error = f"Gemini reply failed: {type(e).__name__}: {e}"
    else:
        st.session_state.gemini_error = "No Gemini client — check that the API key loaded correctly."

    st.rerun()

st.caption("Demo data only. Not a medical device — for self-reflection and portfolio purposes.")