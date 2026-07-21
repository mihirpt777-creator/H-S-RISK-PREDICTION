"""
ONE FILE — Both models, one webpage.
----------------------------------------
Covers BOTH your domains: hardware (hard-drive failure) and software
(code defect prediction). Same setup as before, same three uses:
run locally, deploy publicly, or run on a real server.

SETUP:
  pip install streamlit joblib numpy scikit-learn

RUN:
  streamlit run app.py
"""
import streamlit as st
import joblib
import numpy as np
import subprocess
import json

st.set_page_config(page_title="Failure & Defect Risk Predictor", page_icon="🔧", layout="centered")

HW_RANGES = {
    "Servo5": (0, 127), "Servo10": (0, 500000), "FlyHeight11": (0, 26728),
    "PList": (0, 96), "FlyHeight5": (0, 604), "FlyHeight6": (0, 26728),
}
HW_SMART_ID_MAP = {"Servo5": 5, "Servo10": 10, "FlyHeight11": 11, "PList": 197, "FlyHeight5": 195, "FlyHeight6": 196}

SW_RANGES = {
    "b": (0.0, 2.64), "i": (0.0, 193.06), "d": (0.0, 53.75),
    "total_Opnd": (0.0, 428.0), "uniq_Op": (0.0, 37.0), "loc": (1.0, 288.0),
}
SW_HINTS = {
    "b": "Halstead bugs estimate", "i": "Halstead intelligence", "d": "Halstead difficulty",
    "total_Opnd": "total operands", "uniq_Op": "unique operators", "loc": "lines of code",
}


@st.cache_resource
def load_models():
    hw = joblib.load("hardware_model.pkl")
    sw = joblib.load("software_model.pkl")
    return hw, sw


hw_saved, sw_saved = load_models()

st.title("🔧 Failure & Defect Risk Predictor")
st.caption("Two real trained models, calibrated with conformal prediction — hard-drive failure (68,411 real records) and software defects (2,109 real NASA code modules).")

domain = st.radio("Choose a model:", ["💾 Hardware — Hard Drive Failure", "💻 Software — Code Defect Risk"], horizontal=True)

st.divider()

if domain.startswith("💾"):
    model, feature_names = hw_saved["model"], hw_saved["feature_names"]
    qhat, threshold = hw_saved["conformal_qhat"], hw_saved["optimal_threshold"]

    tab1, tab2 = st.tabs(["🖥️ Scan this computer's real drive", "🎚️ Try your own values"])

    with tab1:
        st.write("Reads this computer's actual drive, live. Needs admin rights and smartmontools installed. Won't work on cloud-hosted versions (no physical drive to read).")
        if st.button("Scan now"):
            try:
                scan = subprocess.run(["smartctl", "--scan"], capture_output=True, text=True)
                lines = [l for l in scan.stdout.splitlines() if l.strip() and not l.startswith("#")]
                if not lines:
                    st.error("No drive found. Run this from an Administrator terminal, on a real computer.")
                else:
                    device = lines[0].split()[0]
                    result = subprocess.run(["smartctl", "-a", "-j", device], capture_output=True, text=True)
                    data = json.loads(result.stdout)
                    st.write(f"**Drive found:** {data.get('model_name', 'Unknown')} ({device})")
                    table = data.get("ata_smart_attributes", {}).get("table", [])
                    attr_by_id = {a["id"]: a.get("raw", {}).get("value", 0) for a in table}
                    mapped_ids = [HW_SMART_ID_MAP[f] for f in feature_names if f in HW_SMART_ID_MAP]
                    found = sum(1 for i in mapped_ids if i in attr_by_id)
                    coverage = found / len(mapped_ids) if mapped_ids else 0
                    x = np.array([attr_by_id.get(HW_SMART_ID_MAP.get(f), 0) for f in feature_names])
                    prob = model.predict_proba([x])[0][1]
                    if coverage < 0.5:
                        st.warning(f"Only {coverage:.0%} of expected attributes found (common for SSDs). Prediction below isn't reliable — try the sliders tab instead.")
                        st.metric("Raw output (low confidence)", f"{prob:.1%}")
                    else:
                        st.metric("Failure risk", f"{prob:.1%}")
                        st.error("⚠️ Needs attention") if prob >= threshold else st.success("✅ Healthy")
            except FileNotFoundError:
                st.error("smartctl not found. Install smartmontools first.")

    with tab2:
        st.write("Move any slider — real trained model, computing live.")
        vals = {}
        cols = st.columns(2)
        for i, (fname, (mn, mx)) in enumerate(HW_RANGES.items()):
            with cols[i % 2]:
                vals[fname] = st.slider(fname, mn, mx, int((mn + mx) * 0.15))
        x = np.array([vals.get(f, 0) for f in feature_names])
        prob = model.predict_proba([x])[0][1]
        st.metric("Failure risk", f"{prob:.1%}")
        st.error("⚠️ Needs attention") if prob >= threshold else st.success("✅ Healthy")
        lo, hi = max(0, prob - qhat), min(1, prob + qhat)
        st.caption(f"90%-confidence range: {lo:.1%} – {hi:.1%}")

else:
    model, feature_names = sw_saved["model"], sw_saved["feature_names"]
    qhat, threshold = sw_saved["conformal_qhat"], sw_saved["optimal_threshold"]

    st.write("Enter code metrics for a module (or paste values from a real static-analysis tool) — real trained model, computing live.")
    vals = {}
    cols = st.columns(2)
    for i, (fname, (mn, mx)) in enumerate(SW_RANGES.items()):
        with cols[i % 2]:
            vals[fname] = st.slider(f"{fname} ({SW_HINTS[fname]})", float(mn), float(mx), float(mn + (mx - mn) * 0.2))

    x = np.array([vals.get(f, 0) for f in feature_names])
    prob = model.predict_proba([x])[0][1]
    st.metric("Defect risk", f"{prob:.1%}")
    st.error("⚠️ Likely defective") if prob >= threshold else st.success("✅ Likely clean")
    lo, hi = max(0, prob - qhat), min(1, prob + qhat)
    st.caption(f"90%-confidence range: {lo:.1%} – {hi:.1%}")

st.divider()
st.caption("Both models are real and trained — every prediction above is computed live from the actual saved model files, not hardcoded.")
