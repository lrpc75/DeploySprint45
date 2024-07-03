import os
import tarfile
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from dotenv import load_dotenv
import os

def load_model():
    # Carregar variáveis de ambiente do arquivo .env
    load_dotenv()

    # Carregar os artefatos do modelo do S3 para um diretório local
    local_model_dir = './src/api/models/'
    #os.makedirs(local_model_dir, exist_ok=True)
    #s3_client.download_file(s3_bucket, s3_key, f'{local_model_dir}/model.tar.gz')

    # Diretório onde o modelo .tar.gz está localizado
    model_tar_gz_path = f'{local_model_dir}/model.tar.gz'

    # Diretório onde você deseja extrair o modelo
    extracted_model_dir = './extracted_model'
    os.makedirs(extracted_model_dir, exist_ok=True)

    # Extrair o modelo .tar.gz
    with tarfile.open(model_tar_gz_path, 'r:gz') as tar:
        tar.extractall(path=extracted_model_dir)

    # Carregar o modelo XGBoost
    model_file_path = os.path.join(extracted_model_dir, 'xgboost-model')
    bst = xgb.Booster()
    bst.load_model(model_file_path)

    # Salvar o modelo localmente usando joblib
    joblib.dump(bst, 'xgboost_model.joblib')

    # Para carregar o modelo posteriormente
    return joblib.load('xgboost_model.joblib')

def preprocess_data(data):
    df = pd.DataFrame([data])

    # Codificar variáveis categóricas
    categorical_columns = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Verificar se todas as colunas necessárias estão presentes no DataFrame
    columns = [
        'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
        'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
        'lead_time', 'arrival_year', 'arrival_month', 'arrival_date',
        'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled', 'no_of_special_requests'
    ]

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"A coluna {col} está faltando nos dados de entrada")

    return df[columns].values
