import boto3
import pandas as pd
from sqlalchemy import create_engine, text
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Cria uma sessão boto3 usando o perfil SSO
boto3_session = boto3.Session(profile_name=os.getenv('PROFILE_NAME'))

# Cria uma sessão SageMaker usando a sessão boto3
sagemaker_session = sagemaker.Session(boto_session=boto3_session)

# Pega a role do SageMaker
role = os.getenv('SAGEMAKER_ROLE')

#Informações para acessar o banco de dados 
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')

# Cria a string de conexão SQLAlchemy
db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

try:
    # Cria a engine de conexão
    engine = create_engine(db_uri)

    # Conecta à base de dados
    connection = engine.connect()

    # Verifica a conexão (opcional)
    if connection:
        print("Conexão bem-sucedida!")

    query = text("SELECT * FROM Hotel_Reservations;")

    # Carrega os dados em um DataFrame do Pandas
    hotel_reservations = pd.read_sql(query, engine)
    print(hotel_reservations.head())

except Exception as e:
    print(f"Erro durante a execução da consulta: {e}")

finally:
    # Fecha a conexão
    if 'connection' in locals():
        connection.close()
        print("Conexão fechada.")

# Pré-processamento dos dados
# Cria a nova coluna 'label_avg_price_per_room'
def categorize_price(price):
    if price <= 85:
        return 1
    elif price < 115:
        return 2
    else:
        return 3

hotel_reservations['label_avg_price_per_room'] = hotel_reservations['avg_price_per_room'].apply(categorize_price)

# Exclui a coluna 'avg_price_per_room'
hotel_reservations.drop('avg_price_per_room', axis=1, inplace=True)

# Codifica variáveis categóricas
label_encoders = {}
categorical_columns = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status']
for column in categorical_columns:
    le = LabelEncoder()
    hotel_reservations[column] = le.fit_transform(hotel_reservations[column])
    label_encoders[column] = le

# Separa características e variável alvo
X = hotel_reservations.drop(['Booking_ID', 'label_avg_price_per_room'], axis=1)
y = hotel_reservations['label_avg_price_per_room']

# Divide os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Converte os dados de treino e teste em DataFrames para SageMaker
train_data = pd.DataFrame(X_train)
train_data['label'] = y_train.values

test_data = pd.DataFrame(X_test)
test_data['label'] = y_test.values

# Salva os conjuntos de dados em arquivos CSV
train_data.to_csv('train.csv', index=False, header=False)
test_data.to_csv('test.csv', index=False, header=False)

# Carrega os dados no Amazon S3
s3 = boto3_session.resource('s3')
bucket = sagemaker_session.default_bucket()
prefix = 'hotel-reservations-xgboost'

# Upload dos arquivos CSV para o S3
train_s3_path = f's3://{bucket}/{prefix}/train/train.csv'
test_s3_path = f's3://{bucket}/{prefix}/test/test.csv'
s3.Bucket(bucket).upload_file('train.csv', f'{prefix}/train/train.csv')
s3.Bucket(bucket).upload_file('test.csv', f'{prefix}/test/test.csv')

# Configuração do treinamento do XGBoost no SageMaker
xgboost_estimator = XGBoost(
    entry_point='train.py',  # Arquivo que contém o código de treinamento
    role=role,
    instance_count=1,
    instance_type='ml.m5.4xlarge',
    framework_version='1.3-1',
    py_version='py3',
    output_path=f's3://{bucket}/{prefix}/output',
    sagemaker_session=sagemaker_session
)

# Definição dos inputs de treinamento
train_input = TrainingInput(s3_data=train_s3_path, content_type='csv')
test_input = TrainingInput(s3_data=test_s3_path, content_type='csv')

# Iniciar o treinamento
xgboost_estimator.fit({'train': train_input, 'validation': test_input})

# Após o treinamento, baixar o modelo treinado para uso local, se necessário
model_artifact = xgboost_estimator.model_data
print(f'Modelo treinado armazenado em: {model_artifact}')
