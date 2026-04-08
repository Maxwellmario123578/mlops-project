import pickle, os, json, logging, boto3
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from datetime import datetime
 
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
 
S3_BUCKET     = os.environ['S3_BUCKET']
MODEL_VERSION = os.environ['MODEL_VERSION']
 
def main():
    # Charger les splits depuis S3
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET,
        f'models/{MODEL_VERSION}/splits.pkl', '/tmp/splits.pkl')
    splits = pickle.load(open('/tmp/splits.pkl', 'rb'))
 
    log.info("Démarrage entraînement RandomForest...")
    model = RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight='balanced',  # gestion du déséquilibre fraude/normal
        n_jobs=-1)
    model.fit(splits['X_train'], splits['y_train'])
 
    # Évaluation
    y_pred  = model.predict(splits['X_test'])
    y_proba = model.predict_proba(splits['X_test'])[:,1]
    auc     = roc_auc_score(splits['y_test'], y_proba)
    f1      = f1_score(splits['y_test'], y_pred)
 
    log.info(f"AUC-ROC: {auc:.4f} | F1-Score: {f1:.4f}")
    log.info(classification_report(splits['y_test'], y_pred))
 
    # Sauvegarder modèle + métriques
    os.makedirs('train/artifacts', exist_ok=True)
    pickle.dump(model, open('train/artifacts/model.pkl', 'wb'))
 
    metrics = {
        'version': MODEL_VERSION,
        'auc_roc': round(auc, 4),
        'f1_score': round(f1, 4),
        'timestamp': datetime.utcnow().isoformat()
    }
    with open('train/artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
 
    # Upload sur S3
    for fname in ['model.pkl', 'metrics.json']:
        s3.upload_file(f'train/artifacts/{fname}',
                       S3_BUCKET, f'models/{MODEL_VERSION}/{fname}')
    log.info(f'Modèle {MODEL_VERSION} sauvegardé sur S3.')
 
if __name__ == '__main__':
    main()
