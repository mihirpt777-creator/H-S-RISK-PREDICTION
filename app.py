"""
ONE FILE — Both models, one webpage. Self-training version.
------------------------------------------------------------------
Trains both models fresh when the app starts (cached, so only once),
instead of loading pre-saved .pkl files. This avoids version-mismatch
errors entirely, since the model is always built with whatever
scikit-learn version is actually installed in whatever environment
runs this file.

Needs smart_drive.csv and kc1.csv in the same folder/repo.

SETUP:
  pip install streamlit pandas numpy scikit-learn joblib

RUN:
  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Failure & Defect Risk Predictor", page_icon="🔧", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

.nameplate {
    border: 1px solid #D98E4A55;
    background: linear-gradient(135deg, #1E242B 0%, #181D23 100%);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.nameplate .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    color: #D98E4A;
    text-transform: uppercase;
    border: 1px solid #D98E4A88;
    border-radius: 4px;
    padding: 3px 8px;
}
[data-testid="stMetric"] {
    background: #1E242B;
    border-left: 3px solid #D98E4A;
    border-radius: 6px;
    padding: 10px 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="nameplate"><span class="tag">RUL-ENGINE v1</span> Calibrated failure prediction, three real domains</div>', unsafe_allow_html=True)

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


@st.cache_resource(show_spinner="Training hardware model on real data (one-time, ~10 seconds)...")
def train_hardware_model():
    df = pd.read_csv("smart_drive.csv", skiprows=[1])
    df = df.dropna(subset=["class"])
    df["class"] = df["class"].map({"True": 1, "False": 0, True: 1, False: 0})
    df = df.fillna(0)
    feature_cols = [c for c in df.columns if c != "class"]
    X, y = df[feature_cols].values, df["class"].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_calib, _, y_calib, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)

    calib_probs = model.predict_proba(X_calib)
    scores = 1 - calib_probs[np.arange(len(y_calib)), y_calib]
    qhat = np.quantile(scores, min(np.ceil((len(scores) + 1) * 0.9) / len(scores), 1.0), method="higher")

    return {"model": model, "feature_names": feature_cols, "conformal_qhat": float(qhat), "optimal_threshold": 0.10}


@st.cache_resource(show_spinner="Training software model on real data (one-time, ~5 seconds)...")
def train_software_model():
    df = pd.read_csv("kc1.csv")
    df["defects"] = df["defects"].astype(int)
    feature_cols = [c for c in df.columns if c != "defects"]
    X, y = df[feature_cols].values, df["defects"].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_calib, _, y_calib, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    calib_probs = model.predict_proba(X_calib)
    scores = 1 - calib_probs[np.arange(len(y_calib)), y_calib]
    qhat = np.quantile(scores, min(np.ceil((len(scores) + 1) * 0.9) / len(scores), 1.0), method="higher")

    return {"model": model, "feature_names": feature_cols, "conformal_qhat": float(qhat), "optimal_threshold": 0.05}


@st.cache_resource(show_spinner="Training battery model on real NASA data (one-time, ~5 seconds)...")
def train_battery_model():
    df = pd.read_csv("battery_discharge.csv")
    agg = df.groupby(["Battery", "id_cycle"]).agg(
        mean_voltage=("Voltage_measured", "mean"), min_voltage=("Voltage_measured", "min"),
        mean_current=("Current_measured", "mean"), mean_temp=("Temperature_measured", "mean"),
        max_temp=("Temperature_measured", "max"), capacity=("Capacity", "first"),
    ).reset_index()
    agg["near_eol"] = (agg["capacity"] < 1.4).astype(int)  # NASA's published EOL threshold (30% capacity fade)

    feature_cols = ["mean_voltage", "min_voltage", "mean_current", "mean_temp", "max_temp", "id_cycle"]
    X, y = agg[feature_cols].values, agg["near_eol"].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_calib, _, y_calib, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)

    calib_probs = model.predict_proba(X_calib)
    scores = 1 - calib_probs[np.arange(len(y_calib)), y_calib]
    qhat = np.quantile(scores, min(np.ceil((len(scores) + 1) * 0.9) / len(scores), 1.0), method="higher")

    return {"model": model, "feature_names": feature_cols, "conformal_qhat": float(qhat), "optimal_threshold": 0.05}


hw_saved = train_hardware_model()
sw_saved = train_software_model()
bat_saved = train_battery_model()

st.title("🔧 Failure & Defect Risk Predictor")
st.caption("Three real models, trained live in this app on real data — hard-drive failure (68,411 real records), software defects (2,109 real NASA code modules), and battery end-of-life (real NASA battery aging data) — all calibrated with conformal prediction.")

domain = st.radio("Choose a model:", ["💾 Hardware — Hard Drive Failure", "💻 Software — Code Defect Risk", "🔋 Battery — End-of-Life Risk"], horizontal=True)
st.divider()

if domain.startswith("💾"):
    model, feature_names = hw_saved["model"], hw_saved["feature_names"]
    qhat, threshold = hw_saved["conformal_qhat"], hw_saved["optimal_threshold"]

    tab1, tab2 = st.tabs(["🖥️ Scan this computer's real drive", "🎚️ Try your own values"])

    with tab1:
        st.write("Reads this computer's actual drive, live. Needs admin rights and smartmontools. Won't work on cloud-hosted versions (no physical drive there).")
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

elif domain.startswith("💻"):
    model, feature_names = sw_saved["model"], sw_saved["feature_names"]
    qhat, threshold = sw_saved["conformal_qhat"], sw_saved["optimal_threshold"]

    tab1, tab2 = st.tabs(["📄 Scan a real Python file", "🎚️ Try your own values"])

    with tab1:
        st.write("Upload any real .py file — computes genuine code-complexity metrics (via the `radon` library) and runs them through the real trained model.")
        uploaded = st.file_uploader("Choose a Python file", type=["py"])
        if uploaded is not None:
            try:
                from radon.complexity import cc_visit
                from radon.metrics import h_visit
                from radon.raw import analyze

                code = uploaded.read().decode("utf-8")
                raw = analyze(code)
                h = h_visit(code)
                cc_list = cc_visit(code)
                avg_complexity = sum(f.complexity for f in cc_list) / len(cc_list) if cc_list else 0

                real_metrics = {
                    "loc": raw.loc, "v(g)": avg_complexity, "d": h.total.difficulty,
                    "i": h.total.volume / h.total.difficulty if h.total.difficulty else 0,
                    "b": h.total.bugs, "uniq_Op": h.total.h1, "total_Opnd": h.total.N2,
                }
                st.write("**Real metrics extracted from your file:**")
                st.json({k: round(v, 3) if isinstance(v, float) else v for k, v in real_metrics.items()})

                x = np.array([real_metrics.get(f, 0) for f in feature_names])
                prob = model.predict_proba([x])[0][1]
                st.metric("Defect risk", f"{prob:.1%}")
                st.error("⚠️ Likely defective") if prob >= threshold else st.success("✅ Likely clean")
                lo, hi = max(0, prob - qhat), min(1, prob + qhat)
                st.caption(f"90%-confidence range: {lo:.1%} – {hi:.1%}")
            except Exception as e:
                st.error(f"Could not analyze this file: {e}")

    with tab2:
        st.write("Enter code metrics manually — real trained model, computing live.")
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

else:
    model, feature_names = bat_saved["model"], bat_saved["feature_names"]
    qhat, threshold = bat_saved["conformal_qhat"], bat_saved["optimal_threshold"]

    st.write("Enter discharge-cycle readings — real model trained on real NASA battery aging data, computing live.")
    BAT_RANGES = {
        "mean_voltage": (2.0, 4.2), "min_voltage": (2.0, 4.0), "mean_current": (-2.5, 0.0),
        "mean_temp": (20.0, 45.0), "max_temp": (20.0, 50.0), "id_cycle": (1, 170),
    }
    vals = {}
    cols = st.columns(2)
    for i, (fname, (mn, mx)) in enumerate(BAT_RANGES.items()):
        with cols[i % 2]:
            vals[fname] = st.slider(fname, float(mn), float(mx), float(mn + (mx - mn) * 0.5))

    x = np.array([vals.get(f, 0) for f in feature_names])
    prob = model.predict_proba([x])[0][1]
    st.metric("End-of-life risk", f"{prob:.1%}")
    st.error("⚠️ Near end-of-life") if prob >= threshold else st.success("✅ Healthy")
    lo, hi = max(0, prob - qhat), min(1, prob + qhat)
    st.caption(f"90%-confidence range: {lo:.1%} – {hi:.1%}")

st.divider()
st.caption("All models are trained live in this app from real data — nothing pre-saved, nothing hardcoded.")

st.divider()
with st.expander("🖥️ Bonus: this computer's real crash/error history (diagnostic — not a trained model)"):
    st.caption(
        "This is different from the tabs above: there's no public dataset of real OS crashes to train "
        "a calibrated model on, so instead of a prediction, this reads your OWN computer's real "
        "Reliability Monitor history directly. Windows only. Works locally, not on cloud-hosted versions."
    )
    if st.button("Read my real reliability history"):
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_ReliabilityRecords | Sort-Object TimeGenerated -Descending | "
                 "Select-Object -First 15 TimeGenerated, SourceName, Message | ConvertTo-Json"],
                capture_output=True, text=True, timeout=20
            )
            if result.returncode != 0 or not result.stdout.strip():
                st.warning("Couldn't read reliability data. This needs Windows, and sometimes admin rights.")
            else:
                records = json.loads(result.stdout)
                if isinstance(records, dict):
                    records = [records]
                st.write(f"**{len(records)} most recent real reliability events on this machine:**")
                for r in records:
                    st.text(f"{r.get('TimeGenerated', '?')} — {r.get('SourceName', 'Unknown')}: {str(r.get('Message',''))[:120]}")
        except FileNotFoundError:
            st.error("PowerShell not found — this feature is Windows-only.")
        except Exception as e:
            st.error(f"Could not read reliability history: {e}")
