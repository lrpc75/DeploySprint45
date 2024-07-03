import boto3
import pandas as pd
import sagemaker
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

# Define o nome do endpoint criado no create_endpoint.py
endpoint_name = 'sagemaker-xgboost-2024-06-21-12-29-36-314'

# Cria o objeto Predictor
predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=CSVSerializer(),
    deserializer=CSVDeserializer()
)

# Lê o arquivo CSV
df = pd.read_csv('train.csv')


# Remover o último campo do DataFrame
df.drop(df.columns[-1], axis=1, inplace=True)

#Extrai a primeira linha do arquivo
first_row = df.iloc[6]


# Converta a primeira linha para uma string CSV
csv_payload = first_row.to_frame().T.to_csv(index=False, header=False)

# Verifica o payload CSV para garantir que os dados estão corretos
print("Payload CSV a ser enviado ao endpoint:")
print(csv_payload)

# Envia a solicitação para o endpoint do SageMaker
try:
    response = predictor.predict(data=csv_payload)
    # Exibe a resposta
    print("Resposta do endpoint:")
    print(response)
except Exception as e:
    print("Erro ao invocar o endpoint:")
    print(e)
