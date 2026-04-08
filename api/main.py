import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import pickle, boto3, os, logging, time
from datetime import datetime

logging.basicConfig(level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)

app = FastAPI(title='MLOps Fraud Detection API', version='2.0.0')

# Métriques Prometheus
PREDICT_COUNT   = Counter("predictions_total", "Total prédictions", ["version", "result"])
PREDICT_LATENCY = Histogram("prediction_latency_seconds", "Latence en secondes")
ERROR_COUNT     = Counter("prediction_errors_total", "Erreurs prédiction")

# Variables d'environnement
MODEL_VERSION = os.environ['MODEL_VERSION']
S3_BUCKET     = os.environ['S3_BUCKET']

# Chargement artefacts
def load_artifacts():
    s3 = boto3.client('s3')
    artifacts = {}
    artifact_files = ['model.pkl', 'scaler.pkl', 'encodings.pkl', 'features.pkl']
    
    os.makedirs("/app/model_cache", exist_ok=True)
    
    for f in artifact_files:
        path = f"/app/model_cache/{f}"
        log.info(f'Téléchargement {f}...')
        s3.download_file(S3_BUCKET, f'models/{MODEL_VERSION}/{f}', path)
        with open(path, 'rb') as f_in:
            artifacts[f.replace('.pkl', '')] = pickle.load(f_in)
    return artifacts

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
encodings = artifacts['encodings']
feature_names = artifacts['features']

class Transaction(BaseModel):
    # Correspond aux colonnes du notebook
    Amount: float
    Transaction_Date: str  # Format YYYY-MM-DD HH:MM:SS
    Merchant_Category: str
    Card_Type: str
    Transaction_Type: str
    Country: str
    Device_Type: str
    Is_International: int
    Is_Chip: int
    Is_Pin_Used: int
    Distance_From_Home: float
    Hour_of_Day: int

@app.post('/predict')
def predict(txn: Transaction):
    start = time.time()
    try:
        # 1. Conversion en DataFrame pour faciliter le mapping
        df_input = pd.DataFrame([txn.dict()])
        
        # 2. Prétraitement comme dans train/preprocess.py
        # Date -> Timestamp
        dt = pd.to_datetime(df_input['Transaction_Date'])
        df_input['Timestamp'] = dt.astype('int64') // 10**9
        df_input = df_input.drop('Transaction_Date', axis=1)
        
        # Encodage fréquentiel
        for col, mapping in encodings.items():
            if col in df_input.columns:
                df_input[f"{col}_encode"] = df_input[col].map(mapping).fillna(0)
                df_input = df_input.drop(col, axis=1)
        
        # 3. Réalignement des colonnes dans le bon ordre
        df_input = df_input[feature_names]
        
        # 4. Scaling
        X_scaled = scaler.transform(df_input)
        
        # 5. Prediction
        pred  = int(model.predict(X_scaled)[0])
        # Calcul de la probabilité de fraude (toujours la classe 1)
        proba_fraud = float(model.predict_proba(X_scaled)[0][1])
        
        result = 'fraud' if pred == 1 else 'normal'
        PREDICT_COUNT.labels(version=MODEL_VERSION, result=result).inc()
        PREDICT_LATENCY.observe(time.time() - start)
        
        log.info(f'Prediction: {result} proba={proba_fraud:.4f}')
        return {
            'prediction': pred,
            'fraud_probability': proba_fraud,
            'result': result,
            'version': MODEL_VERSION
        }
    except Exception as e:
        ERROR_COUNT.inc()
        log.error(f'Erreur: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
def health():
    return {'status': 'ok', 'version': MODEL_VERSION}

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get('/')
def root():
    return {'message': 'MLOps Fraud Detection API', 'version': MODEL_VERSION}
