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

    # 2. Déterminer quel modèle tester (Staging ou Production)
    # On essaie d'abord Staging (pour la promotion)
    current_stage = "Staging"
    model_uri = f"models:/{model_name}/{current_stage}"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"📦 Modèle chargé depuis le stage: {current_stage}")
    except Exception:
        # Si pas de Staging, on vérifie la Production (cas du pipeline Main)
        print(f"ℹ️ Aucun modèle en {current_stage}. Tentative en Production...")
        current_stage = "Production"
        model_uri = f"models:/{model_name}/{current_stage}"
        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"📦 Modèle chargé depuis le stage: {current_stage}")
        except Exception as e:
            print(f"❌ Impossible de trouver un modèle en Staging ou Production: {e}")
            raise

    # 3. Charger les données pour le test
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 4. Calculer l'accuracy
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    print(f"📊 Accuracy du modèle ({current_stage}): {acc}")
    
    # --- LE QUALITY GATE ---
    if acc < 0.8:
        print(f"❌ Échec du Quality Gate: Accuracy insuffisante ({acc}).")
        raise Exception(f"Accuracy too low! Deployment/Promotion rejected.")
    
    print("✅ Quality Gate passé avec succès!")

    # --- PROMOTION (Uniquement si on est en Staging) ---
    if current_stage == "Staging":
        latest_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if latest_versions:
            staging_version = latest_versions[0].version
            print(f"🚀 Promotion de la version {staging_version} vers 'Production'...")
            client.transition_model_version_stage(
                name=model_name,
                version=staging_version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"🏆 Succès! Modèle promu en Production.")
        else:
            print("⚠️ Bizarre : Modèle chargé mais version non trouvée.")
    else:
        print("ℹ️ Le modèle est déjà en Production. Validation Guard Gate réussie.")

if __name__ == "__main__":
    test_and_promote_model()