import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def test_and_promote_model():
    # 1. Config DagsHub
    os.environ['MLFLOW_TRACKING_USERNAME'] = "Yannos95"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "c38f0b2ddfa9b0dce8f328d74fd315a06a986427"
    mlflow.set_tracking_uri("https://dagshub.com/Yannos95/MLOps-Final-Project.mlflow")
    
    client = MlflowClient()
    model_name = "IrisLogisticModel"

    # 2. Charger le modèle qui est actuellement en STAGING
    model_uri = f"models:/{model_name}/Staging"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f" Impossible de charger le modèle en Staging: {e}")
        raise

    # 3. Charger les données pour le test de performance (Quality Gate)
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 4. Exécuter la prédiction et calculer l'accuracy
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    print(f" Accuracy du modèle en Staging: {acc}")
    
    # --- LE QUALITY GATE ---
    if acc < 0.8:
        print("❌ Échec du Quality Gate: Accuracy insuffisante.")
        raise Exception(f"Accuracy too low ({acc}). Promotion rejected!")
    
    print(" Quality Gate passé avec succès!")

    # --- PROMOTION AUTOMATIQUE (Exigence Projet) ---
    # Récupérer le numéro de version exact du modèle qui est en Staging
    latest_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_versions:
        raise Exception("Aucune version trouvée en Staging pour la promotion.")
    
    staging_version = latest_versions[0].version
    
    print(f" Promotion de la version {staging_version} vers le stage 'Production'...")
    
    # Basculer la version vers Production et archiver l'ancienne version de Prod
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f" Succès! Le modèle {model_name} v{staging_version} est maintenant en Production.")

if __name__ == "__main__":
    test_and_promote_model()