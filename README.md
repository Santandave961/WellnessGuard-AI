# WellnessGuard AI

Stress pattern detection from wearable physiological signals + an AI companion
that checks in when a stress episode is flagged.

Applies the same anomaly-detection + explainability approach used in the
fraud/AML projects (IsolationForest + XGBoost + SHAP) to a new domain:
personal wellness instead of financial transactions.

**Live demo:** [wellnessguard-ai.streamlit.app](https://wellnessguard-ai-lbrjcgkl3wagqbhyd5ftn9.streamlit.app)

## How it helps
Most people don't notice they're stressed until it's already affected their
focus, sleep, or mood — the physical signs (rising heart rate, skin
conductance shifts) show up before the person consciously feels it.
WellnessGuard AI closes that gap:

- **Catches it earlier** — flags a stress episode from physiological signals
  in real time, rather than waiting for the person to notice on their own
- **Explains why, not just that** — SHAP attribution means the check-in can
  say *what* changed (e.g. "driven mainly by heart rate"), not just "you seem
  stressed," which makes the alert something a person can actually act on
- **Lowers the barrier to doing something about it** — a warm, low-pressure
  companion message with one small suggestion (a breathing pause, a short
  walk) is easier to act on than a clinical alert or a number on a dashboard
- **Builds a pattern over time** — a single stressful moment isn't
  meaningful, but a log of when and how often episodes happen can help
  someone spot what's actually driving their stress (a meeting block, a time
  of day) rather than guessing

It's explicitly a self-reflection tool, not a diagnostic or medical device —
the goal is awareness and a nudge, not a clinical claim.

## Architecture
- `src/anomaly_utils.py` — feature engineering + dual-track anomaly flagging (IsolationForest + XGBoost), SHAP driver attribution, shared `API_FEATURES` list so training and inference use identical columns
- `src/preprocess_wesad.py` — converts raw WESAD `.pkl` subject files into the CSV format `train.py` expects
- `src/make_synthetic_data.py` — generates synthetic data for smoke-testing the pipeline before real WESAD data is available (rule-based labels — not for reporting real metrics)
- `src/train.py` — training script for the stress classifier
- `src/api.py` — FastAPI inference endpoint, SHAP-driven companion prompt builder, Gemini API call for the check-in message
- `dashboard/app.py` — Streamlit dashboard (stress timeline, signal breakdown, companion chat log, live Gemini-powered check-ins)
- `test_gemini.py` — standalone script to test the Gemini API key/connection directly, outside the app

## Dataset
[WESAD](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) —
wearable stress and affect detection, chest + wrist sensors, 15 subjects
(S2–S17, no S1/S12), lab-induced stress/amusement/baseline/meditation
conditions with self-report labels. Direct download, no request form:
https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx

```bash
python src/preprocess_wesad.py --raw_dir "data/WESAD" --output data/wesad_processed.csv
```

**Citation required for any published work:** Schmidt et al., "Introducing
WESAD, a multimodal dataset for Wearable Stress and Affect Detection," ICMI 2018.

## Setup
```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root (see `.env.example`):
```
GEMINI_API_KEY=your-key-here
```
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
Without a key, the companion falls back to a canned check-in message — the
app still runs fine.

## Run locally
```bash
python src/train.py --data data/wesad_processed.csv --output models/stress_clf.joblib
uvicorn src.api:app --reload          # API on :8000
streamlit run dashboard/app.py        # Dashboard on :8501 (currently uses synthetic demo data for the charts)
```

To smoke-test the training pipeline before WESAD is ready:
```bash
python src/make_synthetic_data.py --output data/wesad_processed.csv
```

To test the Gemini connection in isolation (useful for debugging key/network issues):
```bash
python test_gemini.py
```

## Companion LLM
Uses Google's `google-genai` SDK (`gemini-2.5-flash`). `src/api.py` pulls the
top SHAP-driven stress contributors for each flagged prediction and feeds
them into the companion's prompt, so check-ins reference the actual driver
(e.g. "driven mainly by changes in heart rate") rather than a generic message.
Kept deliberately non-clinical — this is a self-reflection tool, not a
medical device.

## Deployment (Streamlit Cloud)
- Push to GitHub — `.gitignore` excludes `.env`, `__pycache__`, model
  artifacts, and generated CSVs
- Deploy via [share.streamlit.io](https://share.streamlit.io), main file
  path: `dashboard/app.py`
- Add the Gemini key under **App settings → Secrets** in TOML format:
  ```toml
  GEMINI_API_KEY = "your-key-here"
  ```
- No `runtime.txt` needed, unpinned `requirements.txt`
- To deploy the trained model with the FastAPI backend elsewhere, force-add
  it past `.gitignore`: `git add -f models/stress_clf.joblib`

## Model performance
Numbers trained on synthetic data are **not reported here** — a
deterministic label rule makes any resulting ROC-AUC meaningless (trivially
solves the rule that generated it). Real WESAD-trained metrics will be added
once training completes on the actual dataset.

## Ethical/framing notes for portfolio writeups
- Never claim clinical/diagnostic accuracy — frame as "self-reflection" and "pattern awareness"
- Be explicit that WESAD is lab-induced stress, not real-world validated — a fair caveat for interviews
- Don't cite any metrics produced from synthetic/demo data
