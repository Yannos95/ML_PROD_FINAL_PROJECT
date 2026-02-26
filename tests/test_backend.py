import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# --- 3 TESTS UNITAIRES (Mandatory) [cite: 49] ---
def test_health_check():
    """Vérifie que l'API est fonctionnelle."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_prediction_input_validation():
    """Vérifie le rejet des schémas invalides."""
    response = client.post("/predict", json={"wrong_key": 1.0})
    assert response.status_code == 422

def test_root_not_found():
    """Vérifie que les routes inexistantes sont gérées."""
    response = client.get("/")
    assert response.status_code == 404

# --- 2 TESTS D'INTÉGRATION (Mandatory) [cite: 50] ---
def test_mlflow_model_loading():
    """Vérifie la connexion avec le Model Registry MLflow[cite: 61, 64]."""
    from backend.main import model
    assert model is not None

def test_prediction_response_structure():
    """Vérifie que le modèle renvoie une structure de données correcte."""
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    response = client.post("/predict", json=payload)
    assert "prediction" in response.json()

# --- 1 TEST END-TO-END (Mandatory) [cite: 51] ---
def test_full_prediction_flow():
    """Simule une requête utilisateur complète jusqu'à la réponse finale."""
    payload = {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 5.1, "petal_width": 1.8}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1, 2]