from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import pickle, boto3, os, logging, time
 
logging.basicConfig(level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)
 
app = FastAPI(title='MLOps Fraud Detection API', version='1.0.0')
 
# Métriques Prometheus
PREDICT_COUNT   = Counter("predictions_total", "Total prédictions", ["version", "result"])
PREDICT_LATENCY = Histogram("prediction_latency_seconds", "Latence en secondes")
ERROR_COUNT     = Counter("prediction_errors_total", "Erreurs prédiction")
 
# Variables d'environnement (secrets via .env)
MODEL_VERSION = os.environ['MODEL_VERSION']
S3_BUCKET     = os.environ['S3_BUCKET']
 
# Chargement modèle au démarrage
def load_model():
    local = "/app/model_cache/model.pkl"
    s3 = boto3.client('s3')
    log.info(f'Chargement modèle {MODEL_VERSION} depuis S3')
    s3.download_file(S3_BUCKET, f'models/{MODEL_VERSION}/model.pkl', local)
    return pickle.load(open(local, 'rb'))
 
model = load_model()
 
class PredictRequest(BaseModel):
    features: list[float]
 
@app.post('/predict')
def predict(req: PredictRequest):
    start = time.time()
    try:
        pred  = int(model.predict([req.features])[0])
        proba = float(model.predict_proba([req.features])[0][1])
        result = 'fraud' if pred == 1 else 'normal'
        PREDICT_COUNT.labels(version=MODEL_VERSION, result=result).inc()
        PREDICT_LATENCY.observe(time.time() - start)
        log.info(f'Prediction: {result} proba={proba:.4f}')
        return {'prediction': pred, 'fraud_probability': proba,
                'result': result, 'model_version': MODEL_VERSION}
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
