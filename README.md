# 🏙️ Unified Smart City Dashboard
**Integrasi Real-time IoT Statistics & AI-Powered Citizen Complaint Analysis**

Aplikasi dashboard cerdas yang menggabungkan pemantauan data infrastruktur kota (IoT) secara *real-time* dan sistem analisis laporan keluhan warga berbasis **Natural Language Processing (NLP)** menggunakan algoritma **XGBoost**.

---

## 🚀 Fitur Utama
* **IoT Traffic Monitor:** Simulasi pemantauan volume lalu lintas secara dinamis.
* **Energy Consumption Tracking:** Monitoring penggunaan energi (kWh) pada aset kota.
* **Infrastructure Logs:** Laporan status kesehatan aset infrastruktur (Critical/Healthy).
* **AI Complaint Classifier:** Prediksi instansi pemerintah berdasarkan deskripsi keluhan (English).

---

## 🛠️ Tech Stack
* **Backend:** FastAPI (Python 3.x)
* **Machine Learning:** XGBoost, Scikit-Learn (TF-IDF Vectorization)
* **Frontend:** Bootstrap 5, Javascript (Fetch API)
* **Data Handling:** Pandas & Pickle

---

## 📂 Struktur Project
```text
Study-Case-SmartCity/
├── 📁 models/           # Artifacts Machine Learning (.json & .pkl)
├── 📁 src/              # Logic Backend (FastAPI & Datasets)
├── 📁 web/              # Dashboard Frontend (HTML/CSS)
├── requirements.txt     # Daftar Library Python
└── README.md            # Dokumentasi Project