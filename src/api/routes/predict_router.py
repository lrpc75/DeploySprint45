from fastapi import APIRouter, HTTPException
from src.api.models.model import load_model, preprocess_data
from src.api.schemas.schemas import PredictionInput
import xgboost as xgb

# Carregar o modelo treinado
bst = load_model()

# Inicializar o roteador FastAPI
router = APIRouter()

# Definir rota para endpoint de predição
@router.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Converter o schema em um dicionário
        data_dict = data.dict()
        
        # Pré-processar os dados recebidos
        processed_data = preprocess_data(data_dict)
        
        # Converter os dados processados para DMatrix do XGBoost
        dmatrix = xgb.DMatrix(processed_data)

        # Fazer predições com o modelo carregado
        probabilities = bst.predict(dmatrix)
        
        # Encontrar a classe com a maior probabilidade
        predicted_classes = probabilities.argmax(axis=1) + 1 
        
        return {"result": predicted_classes.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
