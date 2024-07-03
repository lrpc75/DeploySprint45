# Use uma imagem base oficial do Python
FROM python:3.9-slim

# Defina o diretório de trabalho no contêiner
WORKDIR /app

# Copie os arquivos de requisitos para o contêiner
COPY requirements.txt .

# Instale as dependências necessárias, incluindo xgboost
RUN pip install --no-cache-dir -r requirements.txt \
&& pip install xgboost 

RUN apt-get update && \
    apt-get install -y curl unzip

# Copie o resto dos arquivos da aplicação para o contêiner
COPY src/ /app/src
COPY .env /app/.env

# Exponha a porta que a aplicação vai rodar
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
