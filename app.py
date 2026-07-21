"""
ONE FILE — Hard Drive Failure Risk, as a real webpage.
--------------------------------------------------------
This replaces the separate demo scripts. Run it once, get a real
webpage (in your browser) that anyone nearby can test through your
laptop's screen -- and later, deploy it for a public link, or run it
on a real server. Same file, three uses.

SETUP (one-time):
  pip install streamlit joblib numpy scikit-learn

RUN (right now, on your laptop):
  streamlit run app.py
  -> opens automatically at http://localhost:8501 in your browser
"""
import streamlit as st
import joblib
import numpy as np
import subprocess
import json

st.set_page_config(page_title="Hard Drive Failure Risk", page_icon="💾", layout="centered")

FEATURE_RANGES = {
    "Servo5": (0, 127), "Servo10": (0, 500000), "FlyHeight11": (0, 26728),
    "PList": (0, 96), "FlyHeight5": (0, 604), "FlyHeight6": (0, 26728),
}
SMART_ID_MAP = {"Servo5": 5, "Servo10": 10, "FlyHeight11": 11, "PList": 197, "FlyHeight5": 195, "FlyHeight6": 196}


@st.cache_resource
def load_model():
    return joblib.load("hardware_model.pkl")


saved = load_model()
model, feature_names = saved["model"], saved["feature_names"]
qhat, threshold = saved["conformal_qhat"], saved["optimal_threshold"]

st.title("💾 Hard Drive Failure Risk Predictor")
st.caption("Real Gradient Boosting model, calibrated with conformal prediction — trained on 68,411 real hard-drive records.")

tab1, tab2 = st.tabs(["🖥️ Scan this computer's real drive", "🎚️ Try your own values"])

with tab1:
    st.write("Reads this computer's actual drive, live. Needs admin rights and smartmontools installed.")
    if st.button("Scan now"):
        try:
            scan = subprocess.run(["smartctl", "--scan"], capture_output=True, text=True)
            lines = [l for l in scan.stdout.splitlines() if l.strip() and not l.startswith("#")]
            if not lines:
                st.error("No drive found. Run 'streamlit run app.py' from an Administrator terminal.")
            else:
                device = lines[0].split()[0]
                result = subprocess.run(["smartctl", "-a", "-j", device], capture_output=True, text=True)
                data = json.loads(result.stdout)
                st.write(f"**Drive found:** {data.get('model_name', 'Unknown')} ({device})")

                table = data.get("ata_smart_attributes", {}).get("table", [])
                attr_by_id = {a["id"]: a.get("raw", {}).get("value", 0) for a in table}
                mapped_ids = [SMART_ID_MAP[f] for f in feature_names if f in SMART_ID_MAP]
                found = sum(1 for i in mapped_ids if i in attr_by_id)
                coverage = found / len(mapped_ids) if mapped_ids else 0

                x = np.array([attr_by_id.get(SMART_ID_MAP.get(f), 0) for f in feature_names])
                prob = model.predict_proba([x])[0][1]

                if coverage < 0.5:
                    st.warning(
                        f"This drive only reports {coverage:.0%} of the attributes this model expects "
                        "(common for SSDs — the model was trained on mechanical hard-drive data). "
                        "The raw prediction below isn't reliable here — try the 'Try your own values' tab instead."
                    )
                    st.metric("Raw model output (low confidence)", f"{prob:.1%}")
                else:
                    st.metric("Failure risk", f"{prob:.1%}")
                    if prob >= threshold:
                        st.error("⚠️ Needs attention")
                    else:
                        st.success("✅ Healthy")
        except FileNotFoundError:
            st.error("smartctl not found — install smartmontools first (winget install smartmontools.smartmontools).")

with tab2:
    st.write("Move any slider — this is the real trained model, computing live, on any values you choose.")
    vals = {}
    cols = st.columns(2)
    for i, (fname, (mn, mx)) in enumerate(FEATURE_RANGES.items()):
        with cols[i % 2]:
            vals[fname] = st.slider(fname, mn, mx, int((mn + mx) * 0.15))

    x = np.array([vals.get(f, 0) for f in feature_names])
    prob = model.predict_proba([x])[0][1]

    st.metric("Failure risk", f"{prob:.1%}")
    if prob >= threshold:
        st.error("⚠️ Needs attention")
    else:
        st.success("✅ Healthy")

    lower_conf = max(0, prob - qhat)
    upper_conf = min(1, prob + qhat)
    st.caption(f"90%-confidence range: {lower_conf:.1%} – {upper_conf:.1%}")

st.divider()
st.caption("Real trained model. Not a demo with fake numbers — every prediction above is computed live from the actual model file (hardware_model.pkl).")
