import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
import boto3, pickle, os, logging

logging.basicConfig(level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)

S3_BUCKET     = os.environ['S3_BUCKET']
MODEL_VERSION = os.environ['MODEL_VERSION']

def replace_outliers(df, target_column):
    cols = df.select_dtypes(include=['number']).columns.drop(target_column)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df

def main():
    s3 = boto3.client('s3')

    # Télécharger le dataset Parquet depuis S3
    os.makedirs('train/data', exist_ok=True)
    log.info("Téléchargement du dataset Parquet depuis S3...")
    s3.download_file(S3_BUCKET,
        'data/credit_card_fraud_2025.parquet',
        'train/data/credit_card_fraud_2025.parquet')

    # Charger
    log.info("Chargement du dataset...")
    data = pd.read_parquet('train/data/credit_card_fraud_2025.parquet')
    log.info(f"Dataset: {len(data)} lignes | Fraudes: {data['Fraud_Flag'].sum()}")

    # Suppression colonnes inutiles
    to_drop = [c for c in ["Transaction_ID", "Customer_ID", "Merchant_ID"] if c in data.columns]
    data = data.drop(to_drop, axis=1)

    # Conversion date en timestamp
    if "Transaction_Date" in data.columns:
        data['Transaction_Date'] = pd.to_datetime(data['Transaction_Date'])
        data['Timestamp'] = data['Transaction_Date'].astype('int64') // 10**9
        data = data.drop("Transaction_Date", axis=1)

    # Encodage fréquentiel (Multiencodage — comme dans le notebook)
    cat_cols = ["Merchant_Category", "Card_Type", "Transaction_Type", "Country", "Device_Type"]
    encoding_maps = {}
    for col in cat_cols:
        if col in data.columns:
            freq = data.groupby(col).size() / len(data)
            encoding_maps[col] = freq.to_dict()
            data[f"{col}_encode"] = data[col].map(freq)
            data = data.drop(col, axis=1)

    # Traitement valeurs aberrantes
    data = replace_outliers(data, 'Fraud_Flag')

    # Séparation features / target
    X = data.drop("Fraud_Flag", axis=1)
    y = data["Fraud_Flag"]
    feature_names = list(X.columns)
    log.info(f"Features ({len(feature_names)}): {feature_names}")

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # SMOTEENN — comme dans le notebook
    log.info("Application SMOTEENN (peut prendre quelques minutes)...")
    smote_enn = SMOTEENN(random_state=42)
    X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_scaled, y_train)
    log.info(f"Après SMOTEENN: {dict(zip(*np.unique(y_train_balanced, return_counts=True)))}")

    # Sauvegarde artefacts
    os.makedirs('train/artifacts', exist_ok=True)
    pickle.dump(scaler,         open('train/artifacts/scaler.pkl', 'wb'))
    pickle.dump(encoding_maps,  open('train/artifacts/encodings.pkl', 'wb'))
    pickle.dump(feature_names,  open('train/artifacts/features.pkl', 'wb'))
    pickle.dump({
        'X_train': X_train_balanced,
        'y_train': y_train_balanced,
        'X_test':  X_test_scaled,
        'y_test':  y_test.values
    }, open('train/artifacts/splits.pkl', 'wb'))

    # Upload S3
    for fname in ['scaler.pkl', 'encodings.pkl', 'features.pkl', 'splits.pkl']:
        s3.upload_file(f'train/artifacts/{fname}',
                       S3_BUCKET, f'models/{MODEL_VERSION}/{fname}')
        log.info(f'Uploadé: models/{MODEL_VERSION}/{fname}')

    log.info('Preprocessing terminé avec succès.')

if __name__ == '__main__':
    main()
