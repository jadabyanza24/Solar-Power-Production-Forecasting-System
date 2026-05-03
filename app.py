import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import time
import io

# Konfigurasi Standar Industri
st.set_page_config(page_title="Photovoltaic Management System", layout="wide")

# Prosedur Inisialisasi Model
@st.cache_resource
def initialize_engine():
    try:
        engine = XGBRegressor()
        # Memuat model yang telah dioptimasi dengan Hyperparameter Tuning
        engine.load_model("model.json")
        return engine
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None

model_engine = initialize_engine()

# Inisialisasi Session State untuk alur formulir
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False
if 'location_data' not in st.session_state:
    st.session_state.location_data = {"lat": -6.2088, "long": 106.8456, "name": "Jakarta Site 01"}

# --- TAHAP 1: FORMULIR SETUP LOKASI ---
if not st.session_state.setup_complete:
    st.title("System Commissioning: Location Setup")
    st.write("Silakan masukkan parameter geografis untuk kalibrasi model sebelum memulai operasional.")
    
    with st.form("setup_form"):
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Station Latitude", value=st.session_state.location_data["lat"], format="%.4f")
        with col2:
            long = st.number_input("Station Longitude", value=st.session_state.location_data["long"], format="%.4f")
        
        station_name = st.text_input("Station Identification Name", value=st.session_state.location_data["name"])
        
        submit_setup = st.form_submit_button("Complete Initialization")
        
        if submit_setup:
            st.session_state.location_data = {"lat": lat, "long": long, "name": station_name}
            st.session_state.setup_complete = True
            st.rerun()

# --- TAHAP 2: DASHBOARD PREDIKSI PROFESIONAL ---
else:
    st.sidebar.header("System Status")
    st.sidebar.write(f"Station: **{st.session_state.location_data['name']}**")
    st.sidebar.write(f"Coordinates: {st.session_state.location_data['lat']}, {st.session_state.location_data['long']}")
    
    if st.sidebar.button("Re-calibrate Location"):
        st.session_state.setup_complete = False
        st.rerun()

    st.title("Photovoltaic Energy Production Forecasting")
    
    # Fitur Baru: Batch Processing via CSV
    st.subheader("Data Integration")
    tabs = st.tabs(["Manual Entry", "Batch Import (CSV)"])

    # TAB 1: MANUAL ENTRY (Sama seperti sebelumnya)
    with tabs[0]:
        col_in, col_out = st.columns([1, 2], gap="large")
        with col_in:
            st.write("Manual Operational Parameters")
            temp = st.number_input("Ambient Temperature (Celsius)", min_value=-15.0, max_value=55.0, value=25.0, step=0.1)
            ghi = st.number_input("Global Horizontal Irradiance (W/m²)", min_value=0.0, max_value=1200.0, value=500.0, step=1.0)
            
            if st.button("Execute Single Forecasting"):
                if model_engine:
                    input_tensor = np.array([[temp, ghi/1000.0, st.session_state.location_data["lat"], st.session_state.location_data["long"]]])
                    start_time = time.perf_counter()
                    prediction = model_engine.predict(input_tensor)[0]
                    prediction = max(0.0, prediction)
                    latency = (time.perf_counter() - start_time) * 1000
                    
                    st.metric("Predicted Output", f"{prediction:.4f} kWh")
                    st.caption(f"Inference Latency: {latency:.6f} ms")

    # TAB 2: BATCH IMPORT & EXPORT
    with tabs[1]:
        st.write("Unggah file CSV yang berisi kolom: `temperature` dan `ghi`.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            # Validasi kolom
            required_columns = ['temperature', 'ghi']
            if all(col in data.columns for col in required_columns):
                st.success("File format validated.")
                
                if st.button("Process Batch Prediction"):
                    # Menyiapkan data batch dengan koordinat statis dari setup
                    data['latitude'] = st.session_state.location_data['lat']
                    data['longitude'] = st.session_state.location_data['long']
                    
                    # Normalisasi GHI untuk input model
                    features = data[['temperature', 'ghi', 'latitude', 'longitude']].copy()
                    features['ghi'] = features['ghi'] / 1000.0
                    
                    # Eksekusi prediksi massal
                    start_batch = time.perf_counter()
                    predictions = model_engine.predict(features.values)
                    data['predicted_power_kwh'] = np.maximum(0, predictions)
                    end_batch = time.perf_counter()
                    
                    st.write(f"Batch processed in {(end_batch - start_batch):.4f} seconds.")
                    st.dataframe(data.head(10))
                    
                    # Fitur Export: Download Result
                    csv_output = data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Prediction Results (CSV)",
                        data=csv_output,
                        file_name=f"prediction_results_{st.session_state.location_data['name']}.csv",
                        mime='text/csv',
                    )
            else:
                st.error(f"Format CSV tidak sesuai. Pastikan kolom berikut tersedia: {required_columns}")

    st.divider()
    st.caption("Research Implementation: Lightweight ML for Renewable Energy | SDG 7 Compliance | 2026")