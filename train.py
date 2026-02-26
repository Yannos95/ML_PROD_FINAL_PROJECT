import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import subprocess
import os

# --- CONFIGURATION DAGSHUB ---
os.environ['MLFLOW_TRACKING_USERNAME'] = "Yannos95"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "c38f0b2ddfa9b0dce8f328d74fd315a06a986427" 
mlflow.set_tracking_uri("https://dagshub.com/Yannos95/MLOps-Final-Project.mlflow")

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_dvc_hash():
    # Récupère le hash du fichier de données via dvc
    return subprocess.check_output(['py', '-m', 'dvc', 'list', '.', 'data/iris.csv', '--dvc-only']).decode('ascii').strip()

# 1. Chargement des données (traquées par DVC)
df = pd.read_csv('data/iris.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. MLflow Experiment
mlflow.set_experiment("Iris_Classification")

with mlflow.start_run():
    # Paramètres 
    params = {"C": 1.0, "solver": "lbfgs", "max_iter": 100}
    mlflow.log_params(params)
    
    # Tags de traçabilité 
    mlflow.set_tag("git_commit", get_git_revision_hash())
    mlflow.set_tag("dvc_data_version", get_dvc_hash())
    
    # Entraînement
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # Métriques [
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Enregistrement dans le Registry 
    mlflow.sklearn.log_model(model, "model", registered_model_name="IrisLogisticModel")

    print(f"Modèle entraîné avec une accuracy de : {accuracy}")
    #