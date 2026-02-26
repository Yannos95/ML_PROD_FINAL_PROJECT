import mlflow.sklearn
import os
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def test_model_performance():
    # Config DagsHub
    os.environ['MLFLOW_TRACKING_USERNAME'] = "Yannos95"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "c38f0b2ddfa9b0dce8f328d74fd315a06a986427"
    mlflow.set_tracking_uri("https://dagshub.com/Yannos95/MLOps-Final-Project.mlflow")
    
    # Charger le modèle de STAGING
    model_uri = "models:/IrisLogisticModel/Staging"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Charger les données pour un test rapide
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Prédire
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    
    print(f"Model Accuracy in Staging: {acc}")
    
    # LA BARRIÈRE (GATE)
    if acc < 0.8:
        raise Exception(f"Accuracy too low ({acc}). Promotion rejected!")
    
    print("Quality Gate Passed! ✅")

if __name__ == "__main__":
    test_model_performance()