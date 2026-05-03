## Usage

1.  Tempatkan file model `xgboost_solar_edge_TUNED.json` di direktori akar proyek.
2.  Jalankan aplikasi Streamlit:
    ```bash
    streamlit run app.py
    ```
3.  Lakukan **Setup Location** (contoh Jakarta: Latitude -6.2088, Longitude 106.8456) pada layar inisialisasi stasiun.
4.  Masukkan parameter temperatur dan GHI secara manual atau unggah file CSV pada dashboard utama untuk pemrosesan data massal.

## Project Affiliation

*   **Lead Developer**: Jad Abyanza Fauzan
*   **Institution**: Binus University - Computer Science (AI Specialization 2028)
*   **Organization**: HIMTI (General Manager of Research and Development)
*   **Research Focus**: Sustainable Development Goals (SDG) 7

## License

This project is licensed under the MITBerikut adalah file `README.md` lengkap dengan format profesional menggunakan *badges* (lencana) teknis dan struktur organisasi yang formal tanpa menggunakan elemen dekoratif.

---
```markdown
# Solar Power Production Forecasting System

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistem estimasi produksi daya fotovoltaik berbasis **Lightweight Machine Learning** yang dirancang untuk mendukung stabilitas jaringan listrik cerdas (*smart grid*) dan implementasi pada perangkat *edge computing*.

## Project Overview

Proyek ini bertujuan untuk mengimplementasikan model prediksi energi terbarukan yang efisien secara komputasi tanpa mengorbankan akurasi. Fokus utama riset ini adalah mendukung **Sustainable Development Goal 7 (Affordable and Clean Energy)** dengan menyediakan sistem pemantauan daya yang cepat dan andal.

### Key Scientific Contributions
*   **Multi-Continental Data Harmonization**: Model dilatih menggunakan dataset yang telah diharmonisasi dari tiga benua (India, USA, dan Australia) untuk menjamin kemampuan generalisasi universal.
*   **Optimized XGBoost Architecture**: Melalui proses *hyperparameter tuning* intensif dengan *5-Fold Cross Validation*, model mencapai titik optimal antara akurasi dan efisiensi energi komputasi.
*   **Low-Latency Edge AI**: Menghasilkan inferensi dalam waktu kurang dari 0.01 ms, menjadikannya ideal untuk respon instan terhadap fluktuasi cuaca pada sistem kontrol *real-time*.

## Technical Performance

Berdasarkan pengujian akhir menggunakan data pengujian yang tidak terlihat (*unseen data*), model menunjukkan performa sebagai berikut:

| Metric | Value |
| :--- | :--- |
| **Algorithm** | Tuned XGBoost Regressor |
| **Test R-Squared** | 0.802263 |
| **Inference Latency** | 0.003535 ms |
| **R-Squared Drop** | 0.012557 (Good Fit) |

Analisis *overfitting check* menunjukkan selisih akurasi antara data *train* dan *test* hanya sebesar **1.2%**, membuktikan kemampuan generalisasi yang sangat kuat.

## Features

1.  **System Commissioning (Setup Wizard)**: Inisialisasi parameter geografis (Latitude & Longitude) untuk kalibrasi spesifik lokasi sebelum operasional dimulai.
2.  **Professional Operational Dashboard**: Antarmuka bersih dengan standar industri, dirancang untuk monitoring teknis profesional.
3.  **Standardized GHI Input**: Menggunakan satuan *Global Horizontal Irradiance* (W/m²) sesuai standar industri meteorologi dan energi surya.
4.  **Batch Processing**: Fitur import data massal melalui CSV untuk analisis data historis dalam jumlah besar.
5.  **Export Result**: Ekspor hasil prediksi kembali ke format CSV untuk pelaporan teknis atau analisis lanjutan.

## Installation

Pastikan sistem Anda telah terinstal Python 3.9 atau versi yang lebih baru.

```bash
# Clone the repository
git clone https://github.com/username/solar-power-forecasting.git

# Install dependencies
pip install streamlit pandas numpy xgboost scikit-learn
