import joblib
import numpy as np
import os
import pytest
from sklearn.metrics import mean_squared_error

# Ruta base de modelos
MODEL_DIR = "app/modelos_finales"

# Inputs sintéticos para cada ventana de volatilidad
TEST_INPUTS = {
    7:  [0.25, 0.28, 0.26, 0.27, 0.29, 0.26, 0.28],  # valores altos
    14: [0.20 + 0.01*i for i in range(14)],           # tendencia creciente
    21: [0.35 - 0.005*i for i in range(21)],          # tendencia decreciente
    28: [0.22 + np.sin(i/5)*0.01 for i in range(14)]  # leve ciclo senoidal
}

# Valores esperados para la predicción de los próximos días
EXPECTED_OUTPUTS = {
    7:  [0.27]*7,
    14: [0.30]*7,
    21: [0.29]*7,
    28: [0.28]*7
}

# Umbrales de tolerancia
RMSE_TOLERANCES = {7: 0.15, 14: 0.2, 21: 0.3, 28: 0.6}
RESIDUAL_TOLERANCES = {7: 0.2, 14: 0.25, 21: 0.35, 28: 0.6}

@pytest.mark.parametrize("volatility_days", [7, 14, 21, 28])
def test_model_file_exists(volatility_days):
    path = os.path.join(MODEL_DIR, f"mejor_modelo_vol{volatility_days}d.joblib")
    assert os.path.exists(path), f"Modelo {volatility_days}d no encontrado"

@pytest.mark.parametrize("volatility_days", [7, 14, 21, 28])
def test_model_prediction_rmse_and_residuals(volatility_days):
    path = os.path.join(MODEL_DIR, f"mejor_modelo_vol{volatility_days}d.joblib")
    model_data = joblib.load(path)
    
    model = model_data["modelo"]
    scaler_x = model_data["scaler_x"]
    scaler_y = model_data["scaler_y"]
    n_steps_input = model_data["parametros"]["n_steps_input"]
    n_steps_forecast = model_data["parametros"]["n_steps_forecast"]

    # Input normal
    X = np.array(TEST_INPUTS[volatility_days]).reshape(1, -1)
    X_scaled = scaler_x.transform(X)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    y_true = np.array(EXPECTED_OUTPUTS[volatility_days])
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    residuals = y_pred - y_true
    max_residual = np.max(np.abs(residuals))

    # Verificaciones
    assert len(y_pred) == n_steps_forecast
    assert all(v >= 0 for v in y_pred)
    assert rmse < RMSE_TOLERANCES[volatility_days], f"RMSE alto para {volatility_days}d: {rmse:.4f}"
    assert max_residual < RESIDUAL_TOLERANCES[volatility_days], f"Residuo grande para {volatility_days}d: {max_residual:.4f}"

@pytest.mark.parametrize("volatility_days", [7, 14, 21, 28])
def test_extreme_values_prediction(volatility_days):
    path = os.path.join(MODEL_DIR, f"mejor_modelo_vol{volatility_days}d.joblib")
    model_data = joblib.load(path)
    
    model = model_data["modelo"]
    scaler_x = model_data["scaler_x"]
    scaler_y = model_data["scaler_y"]
    n_steps_input = model_data["parametros"]["n_steps_input"]

    # Valores extremos
    X_low = np.array([[0.05]*n_steps_input])
    X_high = np.array([[1.5]*n_steps_input])

    for X_test in [X_low, X_high]:
        X_scaled = scaler_x.transform(X_test)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        assert all(v >= 0 for v in y_pred), f"Predicción negativa para {volatility_days}d con valores extremos"

@pytest.mark.parametrize("volatility_days", [7, 14, 21, 28])
def test_parameters_and_scalers(volatility_days):
    path = os.path.join(MODEL_DIR, f"mejor_modelo_vol{volatility_days}d.joblib")
    model_data = joblib.load(path)

    # Parámetros y métricas
    params = model_data["parametros"]
    metrics = model_data["metricas"]
    assert "volatilidad" in params
    assert "lag" in params
    assert "fold" in params
    assert "n_steps_input" in params
    assert "n_steps_forecast" in params
    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "MAPE" in metrics

    # Scalers
    scaler_x = model_data["scaler_x"]
    scaler_y = model_data["scaler_y"]
    assert hasattr(scaler_x, 'mean_') and hasattr(scaler_x, 'scale_')
    assert hasattr(scaler_y, 'mean_') and hasattr(scaler_y, 'scale_')
    assert len(scaler_x.mean_) == params["n_steps_input"]

