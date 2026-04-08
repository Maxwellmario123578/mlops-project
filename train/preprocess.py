import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import boto3, pickle, os, logging, json
from datetime import datetime
 
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
 
S3_BUCKET     = os.environ['S3_BUCKET']
MODEL_VERSION = os.environ['MODEL_VERSION']
 
def main():
    log.info("Chargement du dataset...")
    df = pd.read_csv("/app/data/creditcard.csv")
    log.info(f"Dataset: {len(df)} lignes, {df['Class'].sum()} fraudes")
 
    # Features et cible
    X = df.drop('Class', axis=1)
    y = df['Class']
 
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
 
    # Sauvegarde locale
    os.makedirs('/app/artifacts', exist_ok=True)
    pickle.dump(scaler, open('/app/artifacts/scaler.pkl', 'wb'))
 
    # Sauvegarde des splits
    splits = {'X_train':X_train,'X_test':X_test,
              'y_train':y_train.values,'y_test':y_test.values}
    pickle.dump(splits, open('/app/artifacts/splits.pkl', 'wb'))
 
    # Upload sur S3
    s3 = boto3.client('s3')
    for fname in ['scaler.pkl', 'splits.pkl']:
        s3.upload_file(f'/app/artifacts/{fname}',
                       S3_BUCKET, f'models/{MODEL_VERSION}/{fname}')
        log.info(f'Upload S3: models/{MODEL_VERSION}/{fname}')
 
    log.info('Preprocessing terminé avec succès.')
 
if __name__ == '__main__':
    main()
