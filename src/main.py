import pandas as pd
import pickle
import random
import os
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SETTING PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models")) 

tfidf = None
label_encoder = None
model = None

print(f"\n🔍 Memulai pengecekan di: {MODEL_DIR}")

# --- FUNGSI LOAD ---
try:
    with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    print("✅ 1. TF-IDF: BERHASIL")
except Exception as e:
    print(f"❌ 1. TF-IDF: GAGAL! (Error: {e})")

try:
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    print("✅ 2. Label Encoder: BERHASIL")
except Exception as e:
    print(f"❌ 2. Label Encoder: GAGAL! (Error: {e})")

try:
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, 'smart_city_xgb_model.json'))
    print("✅ 3. Model XGBoost: BERHASIL")
except Exception as e:
    print(f"❌ 3. Model XGBoost: GAGAL! (Error: {e})")

# --- LOAD CSV ---
try:
    df_traffic = pd.read_csv(os.path.join(BASE_DIR, 'iot_traffic.csv'))
    df_energy = pd.read_csv(os.path.join(BASE_DIR, 'energy_logs.csv'))
    df_infra = pd.read_csv(os.path.join(BASE_DIR, 'infra_logs.csv'))
    print("✅ 4. Semua CSV: BERHASIL\n")
except Exception as e:
    print(f"❌ 4. CSV: GAGAL! ({e})")

class ComplaintRequest(BaseModel):
    description: str

@app.get("/city-stats")
async def get_city_stats():
    traffic = df_traffic.sample(1).to_dict(orient='records')[0]
    energy = df_energy.sample(1).to_dict(orient='records')[0]
    infra_logs = df_infra.sample(5).to_dict(orient='records')
    return {
        "traffic": {"street": traffic.get('STREET'), "speed": int(traffic.get('SPEED')), "status": "Lancar" if traffic.get('SPEED') > 30 else "Padat"},
        "energy": {"usage": round(float(energy.get('KWH_USAGE')), 2), "id": energy.get('ID'), "status": energy.get('STATUS')},
        "infra": infra_logs
    }

@app.post("/predict")
async def predict_complaint(req: ComplaintRequest):
    if tfidf is None or model is None or label_encoder is None:
        return {"error": "Ada file model yang gagal dimuat. Cek terminal!"}
    
    text_vector = tfidf.transform([req.description])
    pred_idx = model.predict(text_vector)
    probs = model.predict_proba(text_vector)
    agency = label_encoder.inverse_transform(pred_idx)[0]
    confidence = max(probs[0]) * 100
    return {"predicted_agency": agency, "confidence_score": f"{confidence:.2f}%"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

# 5. LOAD DATASET CSV 
try:
    df_traffic = pd.read_csv(os.path.join(BASE_DIR, 'iot_traffic.csv'))
    df_energy = pd.read_csv(os.path.join(BASE_DIR, 'energy_logs.csv'))
    df_infra = pd.read_csv(os.path.join(BASE_DIR, 'infra_logs.csv'))
    print("✅ Semua Dataset CSV Berhasil Dimuat!")
except Exception as e:
    print(f"❌ ERROR LOAD CSV: {e}")

# Schema Input
class ComplaintRequest(BaseModel):
    description: str

@app.get("/")
async def root():
    return {"status": "Online", "owner": "Nabila"}

@app.get("/city-stats")
async def get_city_stats():
    traffic = df_traffic.sample(1).to_dict(orient='records')[0]
    energy = df_energy.sample(1).to_dict(orient='records')[0]
    infra_logs = df_infra.sample(5).to_dict(orient='records')
    return {
        "traffic": {
            "street": traffic.get('STREET'),
            "speed": int(traffic.get('SPEED')),
            "status": "Lancar" if traffic.get('SPEED') > 30 else "Padat"
        },
        "energy": {"usage": round(float(energy.get('KWH_USAGE')), 2), "id": energy.get('ID'), "status": energy.get('STATUS')},
        "infra": infra_logs
    }

@app.post("/predict")
async def predict_complaint(req: ComplaintRequest):
    if tfidf is None or model is None:
        return {"error": "Model belum siap"}
    
    text_vector = tfidf.transform([req.description])
    pred_idx = model.predict(text_vector)
    probs = model.predict_proba(text_vector)
    agency = label_encoder.inverse_transform(pred_idx)[0]
    confidence = max(probs[0]) * 100
    
    return {"predicted_agency": agency, "confidence_score": f"{confidence:.2f}%"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)