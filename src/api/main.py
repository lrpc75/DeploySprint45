from fastapi import FastAPI
from src.api.routes.predict_router import router  # importar o roteador do arquivo routes.py

# Inicializar o aplicativo FastAPI
app = FastAPI()

# Incluir o roteador
app.include_router(router)

# Executar o aplicativo FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
