from fastapi.testclient import TestClient
from app.api import app
import pytest

client = TestClient(app)

# --- Test para la ruta raíz ---
def test_raiz_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "modelos_disponibles" in data
    assert isinstance(data["modelos_disponibles"], list)
    assert set(data["modelos_disponibles"]) == {
        "Volatilidad 7 días",
        "Volatilidad 14 días",
        "Volatilidad 21 días",
        "Volatilidad 28 días"
    }

# --- Tests para /predecir endpoint ---
def test_predecir_vol7_valido():
    payload = {
        "lags": [0.52, 0.55, 0.51, 0.53, 0.54, 0.52, 0.56],
        "tipo_volatilidad": 7
    }
    response = client.post("/predecir", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "volatilidad_predicha" in data
    assert len(data["volatilidad_predicha"]) == 7
    assert all(v >= 0 for v in data["volatilidad_predicha"])

def test_predecir_vol14_valido():
    payload = {
        "lags": [0.52, 0.54, 0.53, 0.55, 0.52, 0.54, 0.53, 0.51,
                 0.55, 0.52, 0.54, 0.53, 0.52, 0.54],
        "tipo_volatilidad": 14
    }
    response = client.post("/predecir", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["volatilidad_predicha"]) == 7

def test_predecir_vol21_valido():
    payload = {
        "lags": [0.58] * 21,
        "tipo_volatilidad": 21
    }
    response = client.post("/predecir", json=payload)
    assert response.status_code == 200

def test_predecir_vol28_valido():
    payload = {
        "lags": [0.50] * 14,  # 28 días usa 14 lags
        "tipo_volatilidad": 28
    }
    response = client.post("/predecir", json=payload)
    assert response.status_code == 200

def test_predecir_modelo_invalido():
    payload = {
        "lags": [0.5] * 7,
        "tipo_volatilidad": 99
    }
    response = client.post("/predecir", json=payload)
    assert response.status_code == 400

def test_predecir_lags_incorrectos():
    payload = {
        "lags": [0.5, 0.6],
        "tipo_volatilidad": 7
    }
    response = client.post("/predecir", json=payload)
    assert response.status_code == 400

def test_predecir_volatilidad_baja():
    payload = {
        "lags": [0.15, 0.18, 0.16, 0.17, 0.19, 0.16, 0.15],
        "tipo_volatilidad": 7
    }
    response = client.post("/predecir", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert all(v >= 0 for v in data["volatilidad_predicha"])

def test_predecir_volatilidad_alta():
    payload = {
        "lags": [1.15, 1.25, 1.20, 1.18, 1.22, 1.30, 1.28],
        "tipo_volatilidad": 7
    }
    response = client.post("/predecir", json=payload)
    assert response.status_code == 200

# --- Tests para /info_modelo endpoint ---
def test_info_modelo_valido():
    response = client.get("/info_modelo/7")
    assert response.status_code == 200
    data = response.json()
    assert "dias_volatilidad" in data
    assert "parametros" in data
    assert "metricas" in data
    assert data["dias_volatilidad"] == 7

def test_info_modelo_todos_modelos():
    for vol_dias in [7, 14, 21, 28]:
        response = client.get(f"/info_modelo/{vol_dias}")
        assert response.status_code == 200

def test_info_modelo_invalido():
    response = client.get("/info_modelo/999")
    assert response.status_code == 404
