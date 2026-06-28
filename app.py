import streamlit as st

st.set_page_config(
    page_title="Photovoltaic Management System",
    layout="wide"
)

import pandas as pd
import numpy as np
import pickle
import time
import os
import urllib.request

# ─────────────────────────────────────────────────────────────
# Model loading
# pkl = dict: { 'models': {...}, 'scaler': ..., 'max_power': ..., 'features': [...] }
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "solar_models_india.pkl"
MODEL_URL  = "https://huggingface.co/jadabyanza/solar-model/resolve/main/solar_models_india.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

FEATURE_NAMES = ['irradiance', 'temp', 'lag_power_1h', 'hour']

@st.cache_resource
def load_model_bundle():
    try:
        with open(MODEL_PATH, "rb") as f:
            pkg = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None, []

    models    = pkg.get('models', {})
    scaler    = pkg.get('scaler')
    max_power = pkg.get('max_power', 1.0)
    features  = pkg.get('features', FEATURE_NAMES)

    # Pilih XGBoost sebagai model utama
    xgb = models.get('XGBoost') or models.get('xgboost')
    if xgb is None:
        # fallback: ambil model pertama
        xgb = next(iter(models.values()), None)

    return xgb, scaler, max_power, features

def build_feature_row(irradiance, temp, lag_power_norm, hour):
    row = {
        'irradiance':   irradiance,
        'temp':         temp,
        'lag_power_1h': lag_power_norm,
        'hour':         hour,
    }
    return np.array([[row[f] for f in FEATURE_NAMES]])

def predict_power(model, irradiance, temp, hour, lag_norm, max_power):
    X = build_feature_row(irradiance, temp, lag_norm, hour)
    t0 = time.perf_counter()
    power_norm = model.predict(X)[0]
    latency    = (time.perf_counter() - t0) * 1000
    power_out  = float(np.clip(power_norm, 0, None) * max_power)
    return power_out, latency

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "setup_complete":  False,
    "station_name":    "Jakarta Site 01",
    "max_power":       500.0,
    "last_power_norm": 0.0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# STAGE 1 — Station Setup
# ─────────────────────────────────────────────────────────────
if not st.session_state.setup_complete:
    st.title("System Commissioning: Station Setup")
    st.write(
        "Model dilatih menggunakan data sensor real dari pembangkit fotovoltaik di India "
        "(dataset: Kaggle `anikannal/solar-power-generation-data`, Plant 1). "
        "Masukkan nama stasiun dan kapasitas panel untuk memulai."
    )

    with st.form("setup_form"):
        name_input = st.text_input("Station Name", value=st.session_state.station_name)

        max_power_input = st.number_input(
            "Kapasitas Maksimum Panel (kWh)",
            min_value=1.0, max_value=100_000.0,
            value=float(st.session_state.max_power), step=10.0,
            help="Peak output instalasi — dipakai untuk de-normalisasi prediksi."
        )

        st.info(
            "**Training Region: India** \n\n"
            "Model dilatih pada data irradiance dan suhu sensor nyata dari "
            "Plant 1, India. Prediksi menggunakan fitur: "
            "`irradiance`, `temp`, `lag_power_1h`, `hour`."
        )

        submitted = st.form_submit_button("Complete Initialization")
        if submitted:
            st.session_state.station_name   = name_input
            st.session_state.max_power      = max_power_input
            st.session_state.setup_complete = True
            st.rerun()

# ─────────────────────────────────────────────────────────────
# STAGE 2 — Prediction Dashboard
# ─────────────────────────────────────────────────────────────
else:
    xgb_model, scaler, max_power_loaded, feat_names = load_model_bundle()

    cfg_name   = st.session_state.station_name
    cfg_maxpow = st.session_state.max_power

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.header("System Status")
        st.write(f"**Station:** {cfg_name}")
        st.write(f"**Training region:** `India`")
        st.write(f"**Max capacity:** {cfg_maxpow:,.1f} kWh")
        st.write(f"**Max power (model):** {max_power_loaded:,.2f}")
        st.caption("De-normalisasi menggunakan max_power dari data training.")

        st.divider()

        if xgb_model is not None:
            st.success("XGBoost model ready")
        else:
            st.error("Model tidak ditemukan. Pastikan `solar_models_india.pkl` ada.")

        if feat_names:
            with st.expander("Features"):
                st.code("\n".join(feat_names))

        st.divider()
        if st.button("Re-configure Station"):
            st.session_state.setup_complete  = False
            st.session_state.last_power_norm = 0.0
            st.rerun()

    # ── Header ────────────────────────────────────────────────
    st.title("Photovoltaic Energy Production Forecasting")
    st.caption(
        f"**{cfg_name}** · Training region: **India** · "
        f"Capacity: **{cfg_maxpow:,.1f} kWh**"
    )

    if xgb_model is None:
        st.warning("Model belum termuat. Letakkan `solar_models_india.pkl` di direktori yang sama, lalu refresh.")
        st.stop()

    # ── Tabs ──────────────────────────────────────────────────
    tab_manual, tab_batch = st.tabs(["Manual Entry", "Batch Import (CSV)"])

    # ── TAB 1: MANUAL ─────────────────────────────────────────
    with tab_manual:
        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            st.subheader("Input Parameters")
            hour       = st.slider("Jam Operasional", min_value=6, max_value=18, value=12, step=1)
            temp       = st.number_input("Ambient Temperature (°C)", min_value=-15.0, max_value=55.0, value=28.0, step=0.1)
            irradiance = st.number_input("Irradiance (W/m²)", min_value=0.0, max_value=1400.0, value=600.0, step=1.0)
            lag_norm   = st.number_input(
                "Lag Power 1 Jam Lalu (normalized 0–1)",
                min_value=0.0, max_value=1.0,
                value=float(st.session_state.last_power_norm),
                step=0.01, format="%.3f",
                help="Otomatis terisi dari prediksi terakhir."
            )
            run_btn = st.button("Execute Prediction", use_container_width=True)

        with col_out:
            st.subheader("Output")
            if run_btn:
                power_out, latency = predict_power(
                    xgb_model, irradiance, temp, hour, lag_norm, cfg_maxpow
                )
                norm_val = power_out / cfg_maxpow if cfg_maxpow > 0 else 0.0
                st.session_state.last_power_norm = float(np.clip(norm_val, 0, 1))

                eff_pct = norm_val * 100
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Output", f"{power_out:,.4f} kWh")
                m2.metric("Efficiency",       f"{eff_pct:.1f}%")
                m3.metric("Latency",          f"{latency:.4f} ms")
                st.progress(min(eff_pct / 100, 1.0))

                st.divider()
                detail = pd.DataFrame({
                    "Parameter": ["Hour", "Temperature", "Irradiance", "Lag Power"],
                    "Value":     [hour, f"{temp} °C", f"{irradiance} W/m²", f"{lag_norm:.3f}"],
                })
                st.table(detail.set_index("Parameter"))
                st.info(
                    f"Lag power diperbarui ke **{st.session_state.last_power_norm:.4f}** "
                    "untuk prediksi berikutnya."
                )

                st.divider()
                st.subheader("Production Forecast by Hour")
                hours      = list(range(6, 19))
                hour_preds = [predict_power(xgb_model, irradiance, temp, h, lag_norm, cfg_maxpow)[0] for h in hours]
                hour_df    = pd.DataFrame({"Hour": hours, "Predicted Output (kWh)": hour_preds})
                st.line_chart(hour_df.set_index("Hour"), use_container_width=True)

                st.subheader("Temperature Sensitivity")
                temp_range = np.arange(temp - 10, temp + 11, 1)
                temp_preds = [predict_power(xgb_model, irradiance, t, hour, lag_norm, cfg_maxpow)[0] for t in temp_range]
                temp_df    = pd.DataFrame({"Temperature": temp_range, "Output (kWh)": temp_preds})
                st.line_chart(temp_df.set_index("Temperature"), use_container_width=True)

                st.subheader("Irradiance Sensitivity")
                irr_values = np.arange(0, 1401, 100)
                irr_preds  = [predict_power(xgb_model, irr, temp, hour, lag_norm, cfg_maxpow)[0] for irr in irr_values]
                irr_df     = pd.DataFrame({"Irradiance": irr_values, "Output (kWh)": irr_preds})
                st.line_chart(irr_df.set_index("Irradiance"), use_container_width=True)

                peak_hour = hour_df.loc[hour_df["Predicted Output (kWh)"].idxmax()]
                st.success(
                    f"Peak production diperkirakan terjadi pada pukul "
                    f"**{int(peak_hour['Hour'])}:00** "
                    f"dengan output sekitar **{peak_hour['Predicted Output (kWh)']:.2f} kWh**."
                )
            else:
                st.info("Isi parameter di sebelah kiri, lalu tekan **Execute Prediction**.", icon="👈")

    # ── TAB 2: BATCH ──────────────────────────────────────────
    with tab_batch:
        st.subheader("Batch Prediction via CSV")
        st.write(
            "Upload CSV dengan kolom wajib: `temperature`, `irradiance`, `hour`.  \n"
            "Kolom opsional: `lag_power_1h` (default 0)."
        )

        with st.expander("Format CSV yang diharapkan"):
            sample = pd.DataFrame({
                "temperature":  [28.5, 30.1, 32.0],
                "irradiance":   [600.0, 750.0, 820.0],
                "hour":         [8, 10, 12],
                "lag_power_1h": [0.0, 0.12, 0.20],
            })
            st.dataframe(sample, use_container_width=True)
            st.download_button(
                "Download Sample CSV",
                data=sample.to_csv(index=False).encode("utf-8"),
                file_name="sample_input.csv",
                mime="text/csv",
            )

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(data):,} rows** × {len(data.columns)} cols.")
            st.dataframe(data.head(5), use_container_width=True)

            missing = [c for c in ["temperature", "irradiance", "hour"] if c not in data.columns]
            if missing:
                st.error(f"Kolom wajib tidak ditemukan: `{missing}`")
            else:
                st.success("Format CSV valid")

                if st.button("Process Batch Prediction", use_container_width=True):
                    if "lag_power_1h" not in data.columns:
                        data["lag_power_1h"] = 0.0

                    # Filter jam aktif
                    before = len(data)
                    data   = data[(data["hour"] >= 6) & (data["hour"] <= 18)].copy()
                    if before - len(data):
                        st.warning(f"{before - len(data)} baris di luar jam aktif (6–18) dilewati.")

                    # Build feature matrix
                    X_batch = np.array([
                        [row["irradiance"], row["temperature"], row["lag_power_1h"], int(row["hour"])]
                        for _, row in data.iterrows()
                    ])

                    t0          = time.perf_counter()
                    preds_norm  = xgb_model.predict(X_batch)
                    batch_time  = time.perf_counter() - t0

                    preds_kwh = np.clip(preds_norm, 0, None) * cfg_maxpow
                    data["predicted_power_kwh"] = np.round(preds_kwh, 4)
                    data["power_norm"]           = np.round(np.clip(preds_norm, 0, None), 6)

                    st.divider()
                    st.write(f"**{len(data):,} baris** diproses dalam **{batch_time:.4f}s**.")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Total Output", f"{data['predicted_power_kwh'].sum():,.2f} kWh")
                    s2.metric("Avg Output",   f"{data['predicted_power_kwh'].mean():,.4f} kWh")
                    s3.metric("Max Output",   f"{data['predicted_power_kwh'].max():,.4f} kWh")
                    s4.metric("Min Output",   f"{data['predicted_power_kwh'].min():,.4f} kWh")

                    st.dataframe(data.head(20), use_container_width=True)
                    st.download_button(
                        "Download Prediction Results (CSV)",
                        data=data.to_csv(index=False).encode("utf-8"),
                        file_name=f"prediction_{cfg_name.replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    st.divider()
    st.caption(
        "Research Implementation: Lightweight ML for Renewable Energy | SDG 7 | 2026 | "
        "Model: XGBoost · Dataset: Solar Power Generation Data, India (Kaggle)"
    )