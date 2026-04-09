import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import pickle, boto3, os, logging, time

logging.basicConfig(level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)

app = FastAPI(title='MLOps Fraud Detection API', version='2.0.0')

PREDICT_COUNT   = Counter("predictions_total", "Total prédictions", ["version", "result"])
PREDICT_LATENCY = Histogram("prediction_latency_seconds", "Latence en secondes")
ERROR_COUNT     = Counter("prediction_errors_total", "Erreurs prédiction")

MODEL_VERSION = os.environ['MODEL_VERSION']
S3_BUCKET     = os.environ['S3_BUCKET']

def load_artifacts():
    s3 = boto3.client('s3')
    os.makedirs("/app/model_cache", exist_ok=True)
    artifacts = {}
    for f in ['model.pkl', 'scaler.pkl', 'encodings.pkl', 'features.pkl']:
        path = f"/app/model_cache/{f}"
        log.info(f'Téléchargement {f}...')
        s3.download_file(S3_BUCKET, f'models/{MODEL_VERSION}/{f}', path)
        with open(path, 'rb') as f_in:
            artifacts[f.replace('.pkl', '')] = pickle.load(f_in)
    return artifacts

artifacts     = load_artifacts()
model         = artifacts['model']
scaler        = artifacts['scaler']
encodings     = artifacts['encodings']
feature_names = artifacts['features']

class Transaction(BaseModel):
    Amount:              float
    Transaction_Date:    str
    Merchant_Category:   str
    Card_Type:           str
    Transaction_Type:    str
    Country:             str
    Device_Type:         str
    Is_International:    int
    Is_Chip:             int
    Is_Pin_Used:         int
    Distance_From_Home:  float
    Hour_of_Day:         int

@app.post('/predict')
def predict(txn: Transaction):
    start = time.time()
    try:
        df = pd.DataFrame([txn.dict()])

        # Date → Timestamp
        df['Timestamp'] = pd.to_datetime(df['Transaction_Date']).astype('int64') // 10**9
        df = df.drop('Transaction_Date', axis=1)

        # Encodage fréquentiel
        for col, mapping in encodings.items():
            if col in df.columns:
                df[f"{col}_encode"] = df[col].map(mapping).fillna(0)
                df = df.drop(col, axis=1)

        # Réalignement colonnes
        df = df[feature_names].fillna(0)

        # Scaling
        X_scaled = scaler.transform(df)

        # Prédiction XGBoost natif
        D = xgb.DMatrix(X_scaled)
        proba_raw   = model.predict(D)
        pred        = int(np.argmax(proba_raw[0]))
        proba_fraud = float(proba_raw[0][1])

        if np.isnan(proba_fraud):
            proba_fraud = 0.0

        result = 'fraud' if pred == 1 else 'normal'
        PREDICT_COUNT.labels(version=MODEL_VERSION, result=result).inc()
        PREDICT_LATENCY.observe(time.time() - start)
        log.info(f'Prediction: {result} proba={proba_fraud:.4f}')

        return {
            "prediction":  pred,
            "probability": proba_fraud,
            "status":      result,
            "message":     "Activité suspecte détectée !" if pred == 1 else "La transaction semble légitime.",
            "version":     MODEL_VERSION
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
