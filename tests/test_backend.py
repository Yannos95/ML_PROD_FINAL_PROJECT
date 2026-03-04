import pytest
from fastapi.testclient import TestClient
from backend.main import app, validate_iris_input # Assure-toi que ces fonctions existent

client = TestClient(app)

# =================================================================
# 1. TESTS UNITAIRES
# =================================================================

def test_unit_validation_logic_success():
    """Test unitaire : Vérifie que la validation accepte des données correctes."""
    sample_data = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    assert validate_iris_input(sample_data) is True

def test_unit_validation_logic_failure():
    """Test unitaire : Vérifie que la validation rejette des valeurs négatives."""
    invalid_data = {"sepal_length": -5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    assert validate_iris_input(invalid_data) is False

def test_unit_data_format():
    """Test unitaire : Vérifie que le format de sortie attendu est un dictionnaire."""
    from backend.main import get_app_info
    info = get_app_info()
    assert isinstance(info, dict)
    assert info["app_name"] == "Iris-ML-Prod"

# =================================================================
# 2. TESTS D'INTÉGRATION
# =================================================================

def test_integration_health_endpoint():
    """Vérifie que la route Health répond correctement."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_integration_model_is_loaded():
    """Vérifie que le modèle est bien chargé en mémoire au démarrage."""
    from backend.main import model
    assert model is not None
    assert hasattr(model, "predict")

# =================================================================
# 3. TEST END-TO-END 
# =================================================================

def test_e2e_prediction_flow():
    
    payload = {
        "sepal_length": 5.9,
        "sepal_width": 3.0,
        "petal_length": 5.1,
        "petal_width": 1.8
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1, 2] # Classes Iris : Setosa, Versicolor, Virginica
