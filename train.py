import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import subprocess
import os
import yaml

# --- CONFIGURATION DAGSHUB ---
os.environ['MLFLOW_TRACKING_USERNAME'] = "Yannos95"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "c38f0b2ddfa9b0dce8f328d74fd315a06a986427" 
mlflow.set_tracking_uri("https://dagshub.com/Yannos95/MLOps-Final-Project.mlflow")

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return os.getenv("GITHUB_SHA", "unknown") # Fallback pour GitHub Actions

def get_dvc_hash():
    # Méthode plus fiable : lire le hash directement dans le fichier .dvc
    try:
        with open("data/iris.csv.dvc", "r") as f:
            dvc_data = yaml.safe_load(f)
            return dvc_data['outs'][0]['md5']
    except Exception as e:
        return f"error_fetching_dvc_hash: {str(e)}"

# 1. Chargement des données
df = pd.read_csv('data/iris.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MLflow Experiment
mlflow.set_experiment("Iris_Classification")

with mlflow.start_run():
    # Paramètres 
    params = {"C": 1.0, "solver": "lbfgs", "max_iter": 200} # Augmenté max_iter pour la convergence
    mlflow.log_params(params)
    
    # --- TAGS DE TRAÇABILITÉ (Exigence Projet) ---
    mlflow.set_tag("git_commit", get_git_revision_hash())
    mlflow.set_tag("dvc_data_version", get_dvc_hash())
    
    # Entraînement
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # Métriques
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Signature du modèle (Bonne pratique MLOps pour FastAPI)
    signature = infer_signature(X_test, model.predict(X_test))
    
    # Enregistrement dans le Registry 
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        registered_model_name="IrisLogisticModel",
        signature=signature
    )

    print(f" Modèle entraîné et enregistré. Accuracy : {accuracy}")
    print(f" Git Hash : {get_git_revision_hash()}")
    print(f" DVC Hash : {get_dvc_hash()}")