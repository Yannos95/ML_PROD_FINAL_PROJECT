from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(title="Iris Prediction Service")

# Fonctions isolées pour les TESTS UNITAIRES
def validate_iris_input(data_dict: dict):
    """Logique de validation isolée (Unit Testable)"""
    required_keys = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    if not all(k in data_dict for k in required_keys):
        return False
    # Exemple : rejeter les valeurs négatives
    if any(v < 0 for v in data_dict.values()):
        return False
    return True

def get_app_info():
    """Information système isolée (Unit Testable)"""
    return {"app_name": "Iris-ML-Prod", "stage": "Production"}

# Configuration DagsHub
os.environ['MLFLOW_TRACKING_USERNAME'] = "Yannos95"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "c38f0b2ddfa9b0dce8f328d74fd315a06a986427"
mlflow.set_tracking_uri("https://dagshub.com/Yannos95/MLOps-Final-Project.mlflow")

MODEL_NAME = "IrisLogisticModel"
MODEL_STAGE = "Production"
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

try:
    model = mlflow.sklearn.load_model(model_uri)
except Exception as e:
    model = None

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Appel de la validation (utilisé aussi en test unitaire)
    if not validate_iris_input(data.dict()):
        raise HTTPException(status_code=400, detail="Invalid data")

    input_df = pd.DataFrame([data.dict().values()], 
                            columns=['sepal length (cm)', 'sepal width (cm)', 
                                     'petal length (cm)', 'petal width (cm)'])
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}

@app.get("/health")
def health():
    return {"status": "healthy"}
