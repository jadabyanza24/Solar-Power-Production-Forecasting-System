import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import math

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Photovoltaic Management System", layout="wide")

# ─────────────────────────────────────────────────────────────
# Region assignment — tropical override + bbox-first + nearest edge
#
# Prioritas:
# 1. Zona tropis Asia (lat -15..25, lon 90..145) → India
#    Indonesia, SEA, Filipina: iklim solar paling mirip dataset India.
# 2. Di dalam bbox region → assign langsung.
# 3. Di luar semua bbox → nearest bbox edge (bukan centroid).
# ─────────────────────────────────────────────────────────────

# (lat_min, lat_max, lon_min, lon_max)
REGION_BBOX = {
    "India":     (6.0,   37.0,  67.0,  98.0),
    "Australia": (-44.0, -10.0, 112.0, 154.0),
    "USA":       (24.0,   50.0, -125.0, -65.0),
}

REGION_CENTROIDS = {
    "India":     (20.5937,  78.9629),
    "Australia": (-25.2744, 133.7751),
    "USA":       (37.0902, -95.7129),
}

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _dist_to_bbox(lat, lon, lat_min, lat_max, lon_min, lon_max) -> float:
    """Haversine ke titik terdekat di tepi bbox. 0 jika di dalam."""
    clat = max(lat_min, min(lat, lat_max))
    clon = max(lon_min, min(lon, lon_max))
    return haversine(lat, lon, clat, clon)

def assign_region(lat: float, lon: float) -> str:
    # Climate-aware regional matching
    # Combines geographic distance and latitude similarity.
    scores = {}

    for region in REGION_BBOX:
        geo_dist = _dist_to_bbox(lat, lon, *REGION_BBOX[region])

        region_lat = abs(REGION_CENTROIDS[region][0])
        lat_diff = abs(abs(lat) - region_lat)

        # 1 degree latitude ~= 111 km
        climate_penalty = lat_diff * 300

        scores[region] = geo_dist + climate_penalty

    return min(scores, key=scores.get)

def region_distances(lat: float, lon: float) -> dict:
    return {r: round(haversine(lat, lon, *REGION_CENTROIDS[r])) for r in REGION_CENTROIDS}

def region_bbox_distances(lat: float, lon: float) -> dict:
    return {r: round(_dist_to_bbox(lat, lon, *REGION_BBOX[r])) for r in REGION_BBOX}

# ─────────────────────────────────────────────────────────────
# Feature order (harus match X.columns dari notebook)
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "temp", "irradiance", "lag_power_1h", "hour",
    "location_Australia", "location_India", "location_USA",
]

def build_feature_row(temp, irradiance_wm2, hour, region, lag_power_norm=0.0):
    row = {
        "temp":               temp,
        "irradiance":         irradiance_wm2,
        "lag_power_1h":       lag_power_norm,
        "hour":               hour,
        "location_Australia": 1 if region == "Australia" else 0,
        "location_India":     1 if region == "India"     else 0,
        "location_USA":       1 if region == "USA"       else 0,
    }
    return np.array([[row[f] for f in FEATURE_NAMES]])

# ─────────────────────────────────────────────────────────────
# Load model bundle
# pkl = list of dicts: { 'Model': str, 'Object': trained_model, ... }
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_bundle(pkl_path="model.pkl"):
    try:
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)
    except FileNotFoundError:
        return None, []

    xgb_model = None
    for r in results:
        name = r.get("Model", "")
        obj  = r.get("Object")
        if obj and ("XGBoost" in name or "xgb" in name.lower()):
            xgb_model = obj
            break

    feat_names = []
    if xgb_model is not None:
        try:
            feat_names = xgb_model.get_booster().feature_names or []
        except Exception:
            pass

    return xgb_model, feat_names

def predict_power(model, temp, irradiance_wm2, hour, region, lag_norm, max_power):
    X = build_feature_row(temp, irradiance_wm2, hour, region, lag_norm)
    t0 = time.perf_counter()
    power_norm = model.predict(X)[0]
    latency    = (time.perf_counter() - t0) * 1000
    power_kwh  = float(np.clip(power_norm, 0, None) * max_power)
    return power_kwh, latency

# ─────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "setup_complete":    False,
    "station_name":      "Jakarta Site 01",
    "station_lat":       -6.2088,
    "station_lon":       106.8456,
    "assigned_region":   None,
    "max_power":         500.0,
    "last_power_norm":   0.0,
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
        "Masukkan koordinat stasiun. Sistem akan memilih region dataset "
        "menggunakan Climate-Aware Regional Matching yang mempertimbangkan "
        "kedekatan geografis dan kemiripan karakteristik lintang."
    )

    with st.form("setup_form"):
        name_input = st.text_input("Station Name", value=st.session_state.station_name)
        c1, c2 = st.columns(2)
        with c1:
            lat_input = st.number_input(
                "Latitude", value=st.session_state.station_lat,
                min_value=-90.0, max_value=90.0, format="%.4f",
                help="Positif = utara, negatif = selatan."
            )
        with c2:
            lon_input = st.number_input(
                "Longitude", value=st.session_state.station_lon,
                min_value=-180.0, max_value=180.0, format="%.4f",
                help="Positif = timur Greenwich, negatif = barat."
            )

        max_power_input = st.number_input(
            "Kapasitas Maksimum Panel (kWh)",
            min_value=1.0, max_value=100_000.0,
            value=float(st.session_state.max_power), step=10.0,
            help="Peak output instalasi — dipakai untuk de-normalisasi prediksi."
        )

        # Live preview region assignment sebelum submit
        preview_region  = assign_region(lat_input, lon_input)
        bbox_dists      = region_bbox_distances(lat_input, lon_input)
        centroid_dists  = region_distances(lat_input, lon_input)

        # Tentukan apakah koordinat di dalam bbox
        inside_bbox = bbox_dists[preview_region] == 0
        reason = "koordinat berada di dalam bounding box region ini" if inside_bbox \
            else f"tepi bbox terdekat: {bbox_dists[preview_region]:,} km"

        bbox_detail = "  |  ".join(
            f"{'✅ ' if r == preview_region else ''}{r}: {d:,} km ke bbox"
            for r, d in bbox_dists.items()
        )

        st.info(
            f"**Recommended Training Region: `{preview_region}`**\n\n"
            "Pemilihan dilakukan menggunakan kombinasi "
            "jarak geografis dan kemiripan zona lintang "
            "(Climate-Aware Regional Matching)."
        )

        submitted = st.form_submit_button("Complete Initialization")
        if submitted:
            st.session_state.station_name    = name_input
            st.session_state.station_lat     = lat_input
            st.session_state.station_lon     = lon_input
            st.session_state.assigned_region = preview_region
            st.session_state.max_power       = max_power_input
            st.session_state.setup_complete  = True
            st.rerun()

# ─────────────────────────────────────────────────────────────
# STAGE 2 — Prediction Dashboard
# ─────────────────────────────────────────────────────────────
else:
    xgb_model, feat_names = load_model_bundle()
    cfg_name   = st.session_state.station_name
    cfg_lat    = st.session_state.station_lat
    cfg_lon    = st.session_state.station_lon
    cfg_region = st.session_state.assigned_region
    cfg_maxpow = st.session_state.max_power

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.header("System Status")
        st.write(f"**Station:** {cfg_name}")
        st.write(f"**Coordinates:** {cfg_lat:.4f}, {cfg_lon:.4f}")
        st.write(f"**Assigned region:** `{cfg_region}`")
        st.caption(
            "Region dipilih menggunakan Climate-Aware Regional Matching "
            "(geographic proximity + latitude similarity)."
        )
        st.write(f"**Max capacity:** {cfg_maxpow:,.1f} kWh")

        st.divider()
        st.caption("Distance to region bbox edge (0 = inside)")
        bbox_dists_sb = region_bbox_distances(cfg_lat, cfg_lon)
        for r, d in bbox_dists_sb.items():
            badge = "✅" if r == cfg_region else "  "
            label = "inside bbox" if d == 0 else f"{d:,} km to edge"
            st.write(f"{badge} {r}: **{label}**")

        if xgb_model is not None:
            st.divider()
            st.success("XGBoost model ready")
        else:
            st.error(
                "Model tidak ditemukan. Pastikan `all_training_results_combined.pkl` "
                "ada di direktori yang sama dengan `app.py`."
            )

        if feat_names:
            with st.expander("Feature names (dari booster)"):
                st.code("\n".join(feat_names))

        st.divider()
        if st.button("Re-configure Station"):
            st.session_state.setup_complete  = False
            st.session_state.last_power_norm = 0.0
            st.rerun()

    # ── Header ────────────────────────────────────────────────
    st.title("Photovoltaic Energy Production Forecasting")
    st.caption(
        f"**{cfg_name}** · {cfg_lat:.4f}, {cfg_lon:.4f} "
        f"· Region: **{cfg_region}** · Capacity: **{cfg_maxpow:,.1f} kWh**"
    )

    if xgb_model is None:
        st.warning(
            "Model belum termuat. Letakkan `all_training_results_combined.pkl` "
            "di direktori yang sama dengan `app.py`, lalu refresh."
        )
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
                power_kwh, latency = predict_power(
                    xgb_model, temp, irradiance, hour,
                    cfg_region, lag_norm, cfg_maxpow
                )
                norm_val = power_kwh / cfg_maxpow if cfg_maxpow > 0 else 0.0
                st.session_state.last_power_norm = float(np.clip(norm_val, 0, 1))

                eff_pct = norm_val * 100
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Output", f"{power_kwh:,.4f} kWh")
                m2.metric("Efficiency", f"{eff_pct:.1f}%")
                m3.metric("Latency", f"{latency:.4f} ms")
                st.progress(min(eff_pct / 100, 1.0))

                st.divider()
                detail = pd.DataFrame({
                    "Parameter": ["Hour", "Temperature", "Irradiance", "Lag Power", "Region (auto-assigned)"],
                    "Value":     [hour, f"{temp} °C", f"{irradiance} W/m²", f"{lag_norm:.3f}", cfg_region],
                })
                st.table(detail.set_index("Parameter"))
                st.info(
                    f"Lag power diperbarui ke **{st.session_state.last_power_norm:.4f}** "
                    "untuk prediksi berikutnya."
                )
            else:
                st.info("Isi parameter di sebelah kiri, lalu tekan **Execute Prediction**.", icon="👈")

    # ── TAB 2: BATCH ──────────────────────────────────────────
    with tab_batch:
        st.subheader("Batch Prediction via CSV")
        st.write(
            "Upload CSV dengan kolom wajib: `temperature`, `irradiance`, `hour`.  \n"
            "Kolom opsional: `lag_power_1h` (default 0), `lat` + `lon` (override region per-baris)."
        )

        with st.expander("Format CSV yang diharapkan"):
            sample = pd.DataFrame({
                "temperature":  [28.5, 30.1, 32.0],
                "irradiance":   [600.0, 750.0, 820.0],
                "hour":         [8, 10, 12],
                "lag_power_1h": [0.0, 0.12, 0.20],
                "lat":          [-6.21, -7.25, -6.90],
                "lon":          [106.84, 112.75, 107.60],
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

                    # Tentukan region per baris
                    has_coords = "lat" in data.columns and "lon" in data.columns
                    if has_coords:
                        data["_region"] = data.apply(
                            lambda r: assign_region(r["lat"], r["lon"]), axis=1
                        )
                        region_counts = data["_region"].value_counts().to_dict()
                        st.info(
                            "Region per-baris dari koordinat (bbox-first): "
                            + ", ".join(f"**{r}**: {n} baris" for r, n in region_counts.items()),
                            icon="🌏"
                        )
                    else:
                        data["_region"] = cfg_region
                        st.info(
                            f"Kolom `lat`/`lon` tidak ada — semua baris pakai region setup: **{cfg_region}**",
                            icon="ℹ️"
                        )

                    # Filter jam aktif (sesuai notebook: hour 6–18)
                    before = len(data)
                    data = data[(data["hour"] >= 6) & (data["hour"] <= 18)].copy()
                    skipped = before - len(data)
                    if skipped:
                        st.warning(f"{skipped} baris di luar jam aktif (6–18) dilewati.")

                    # Build feature matrix
                    rows = []
                    for _, row in data.iterrows():
                        feat = {
                            "temp":               row["temperature"],
                            "irradiance":         row["irradiance"],
                            "lag_power_1h":       row["lag_power_1h"],
                            "hour":               int(row["hour"]),
                            "location_Australia": 1 if row["_region"] == "Australia" else 0,
                            "location_India":     1 if row["_region"] == "India"     else 0,
                            "location_USA":       1 if row["_region"] == "USA"       else 0,
                        }
                        rows.append([feat[f] for f in FEATURE_NAMES])

                    X_batch = np.array(rows)
                    t0 = time.perf_counter()
                    preds_norm = xgb_model.predict(X_batch)
                    batch_time = time.perf_counter() - t0

                    preds_kwh = np.clip(preds_norm, 0, None) * cfg_maxpow
                    data["predicted_power_kwh"] = np.round(preds_kwh, 4)
                    data["power_norm"]          = np.round(np.clip(preds_norm, 0, None), 6)
                    data["assigned_region"]     = data["_region"]
                    data = data.drop(columns=["_region"])

                    # Summary metrics
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
        "Model: XGBoost"
    )
