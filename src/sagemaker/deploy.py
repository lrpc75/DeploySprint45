import boto3
import sagemaker
import pandas as pd
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações da sessão
boto3_session = boto3.Session(profile_name=os.getenv('PROFILE_NAME'))
sagemaker_session = sagemaker.Session(boto_session=boto3_session)
role = os.getenv('SAGEMAKER_ROLE')

client = boto3_session.client(service_name = 'sagemaker-runtime', region_name = 'us-east-1')

# Define a URI do artefato do modelo treinado
model_artifact = 's3://sagemaker-us-east-1-654654510951/hotel-reservations-xgboost/output/sagemaker-xgboost-2024-06-20-17-30-14-793/output/model.tar.gz'  

# Criação do modelo no SageMaker
xgboost_model = Model(
    model_data=model_artifact,
    image_uri='683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.3-1',
    role=role,
    sagemaker_session=sagemaker_session
)

# Deploy do modelo como um endpoint
print("Iniciando o deploy do modelo como endpoint...")
try:
    predictor = xgboost_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer()
    )
    print("Endpoint criado com sucesso.")
    print(predictor)
except Exception as e:
    print(f"Erro ao criar o endpoint: {e}")
    predictor = None


