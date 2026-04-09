import pickle, os, json, logging, boto3
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from datetime import datetime

logging.basicConfig(level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)

S3_BUCKET     = os.environ['S3_BUCKET']
MODEL_VERSION = os.environ['MODEL_VERSION']

def main():
    s3 = boto3.client('s3')

    # Charger les splits depuis S3
    log.info("Téléchargement des splits depuis S3...")
    s3.download_file(S3_BUCKET,
        f'models/{MODEL_VERSION}/splits.pkl', '/tmp/splits.pkl')
    splits = pickle.load(open('/tmp/splits.pkl', 'rb'))

    X_train = splits['X_train']
    y_train = splits['y_train']
    X_test  = splits['X_test']
    y_test  = splits['y_test']

    log.info(f"Entraînement XGBoost sur {len(X_train)} exemples...")

    # Entraînement XGBoost — paramètres du notebook
    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test  = xgb.DMatrix(X_test,  label=y_test)

    param = {
        'eta':        0.3,
        'max_depth':  3,
        'objective':  'multi:softprob',
        'num_class':  2
    }

    classifier = xgb.train(param, D_train, 20)

    # Évaluation
    y_proba_raw   = classifier.predict(D_test)
    y_pred        = np.argmax(y_proba_raw, axis=1)
    y_proba_fraud = y_proba_raw[:, 1]

    auc = roc_auc_score(y_test, y_proba_fraud)
    f1  = f1_score(y_test, y_pred)

    log.info(f"AUC-ROC: {auc:.4f} | F1-Score: {f1:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred)}")

    # Sauvegarde modèle
    os.makedirs('train/artifacts', exist_ok=True)
    pickle.dump(classifier, open('train/artifacts/model.pkl', 'wb'))

    metrics = {
        'version':   MODEL_VERSION,
        'auc_roc':   round(auc, 4),
        'f1_score':  round(f1, 4),
        'timestamp': datetime.utcnow().isoformat()
    }
    with open('train/artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Upload S3
    for fname in ['model.pkl', 'metrics.json']:
        s3.upload_file(f'train/artifacts/{fname}',
                       S3_BUCKET, f'models/{MODEL_VERSION}/{fname}')
        log.info(f'Uploadé: models/{MODEL_VERSION}/{fname}')

    log.info(f'Modèle {MODEL_VERSION} sauvegardé sur S3.')

if __name__ == '__main__':
    main()
