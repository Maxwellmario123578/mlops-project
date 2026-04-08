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

def replace_outliers(df, target_column):
    # On sélectionne les colonnes numériques SAUF la target
    cols_to_process = df.select_dtypes(include=['number']).columns.drop(target_column)
    
    for col in cols_to_process:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR  
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    return df

def main():
    log.info("Chargement du dataset...")
    # Le notebook utilise credit_card_fraud_2025.csv, on adapte le chemin
    data_path = "train/data/credit_card_fraud_2025.csv"
    if not os.path.exists(data_path):
        data_path = "train/data/creditcard.csv" # Fallback
        
    df = pd.read_csv(data_path)
    log.info(f"Dataset chargé: {len(df)} lignes")

    # Suppression colonnes inutiles comme dans le notebook
    to_drop = ["Transaction_ID", "Customer_ID", "Merchant_ID"]
    df = df.drop([c for c in to_drop if c in df.columns], axis=1)

    # Conversion date en timestamp
    if "Transaction_Date" in df.columns:
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
        df['Timestamp'] = df['Transaction_Date'].astype('int64') // 10**9
        df = df.drop("Transaction_Date", axis=1)

    # Encodage fréquentiel (Multiencodage du notebook)
    categorical_cols = ["Merchant_Category", "Card_Type", "Transaction_Type", "Country", "Device_Type"]
    encoding_maps = {}
    
    for col in categorical_cols:
        if col in df.columns:
            new_map = df.groupby(col).size() / len(df)
            encoding_maps[col] = new_map.to_dict()
            df[f"{col}_encode"] = df[col].map(new_map)
            df = df.drop(col, axis=1)

    # Traitement des valeurs aberrantes
    target = 'Fraud_Flag' if 'Fraud_Flag' in df.columns else 'Class'
    df = replace_outliers(df, target)

    # Features et cible
    X = df.drop(target, axis=1)
    y = df[target]

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Sauvegarde locale
    os.makedirs('train/artifacts', exist_ok=True)
    pickle.dump(scaler, open('train/artifacts/scaler.pkl', 'wb'))
    pickle.dump(encoding_maps, open('train/artifacts/encodings.pkl', 'wb'))
    
    # On sauve les noms de colonnes pour l'API
    pickle.dump(X.columns.tolist(), open('train/artifacts/features.pkl', 'wb'))

    # Sauvegarde des splits
    splits = {'X_train':X_train,'X_test':X_test,
              'y_train':y_train.values,'y_test':y_test.values}
    pickle.dump(splits, open('train/artifacts/splits.pkl', 'wb'))

    # Upload sur S3
    s3 = boto3.client('s3')
    for fname in ['scaler.pkl', 'encodings.pkl', 'features.pkl', 'splits.pkl']:
        s3.upload_file(f'train/artifacts/{fname}',
                       S3_BUCKET, f'models/{MODEL_VERSION}/{fname}')
        log.info(f'Upload S3: models/{MODEL_VERSION}/{fname}')

    log.info('Preprocessing terminé avec succès.')

if __name__ == '__main__':
    main()
