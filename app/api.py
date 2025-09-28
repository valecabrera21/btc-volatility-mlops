# Librerías necesarias
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os

# Importar schemas
from app.schemas import DatosBTC, RespuestaPrediccion, InfoModelo

# Inicializar FastAPI
app = FastAPI(
    title="API de Predicción de Volatilidad BTC",
    description="Predicciones de volatilidad de Bitcoin usando modelos MLP entrenados por ventana de días.",
    version="1.0.0"
)


# --- Función helper para convertir tipos numpy a tipos nativos de Python ---
def convertir_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convertir_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convertir_numpy(item) for item in obj]
    return obj


# --- Cargar modelos ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelos = {}

for vol in [7, 14, 21, 28]:
    model_path = os.path.join(BASE_DIR, 'modelos_finales', f'mejor_modelo_vol{vol}d.joblib')
    if os.path.exists(model_path):
        modelos[vol] = joblib.load(model_path)
        print(f"✓ Modelo volatilidad {vol} días cargado")
    else:
        print(f"✗ No encontrado: {model_path}")


# --- Ruta raíz ---
@app.get("/", summary="Raíz de la API", tags=["General"])
def raiz():
    return {
        "mensaje": "API de Predicción de Volatilidad BTC",
        "modelos_disponibles": [f"Volatilidad {v} días" for v in modelos.keys()]
    }


# --- Endpoint de predicción ---
@app.post("/predecir", response_model=RespuestaPrediccion, summary="Predicción de volatilidad", tags=["Predicción"])
def predecir(data: DatosBTC):
    if data.tipo_volatilidad not in modelos:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo para {data.tipo_volatilidad} días no disponible. Modelos disponibles: {list(modelos.keys())}"
        )

    modelo_data = modelos[data.tipo_volatilidad]
    modelo = modelo_data['modelo']
    scaler_x = modelo_data['scaler_x']
    scaler_y = modelo_data['scaler_y']
    n_steps_input = modelo_data['parametros']['n_steps_input']
    n_steps_forecast = modelo_data['parametros']['n_steps_forecast']

    if len(data.lags) != n_steps_input:
        raise HTTPException(
            status_code=400,
            detail=f"Se requieren exactamente {n_steps_input} valores de lags. Recibidos: {len(data.lags)}"
        )

    # Preparar datos y escalar
    features = np.array(data.lags).reshape(1, -1)
    features_scaled = scaler_x.transform(features)

    # Predicción
    prediction_scaled = modelo.predict(features_scaled)
    predictions = scaler_y.inverse_transform(prediction_scaled)[0]

    return RespuestaPrediccion(
        tipo_volatilidad=int(data.tipo_volatilidad),
        volatilidad_predicha=convertir_numpy(predictions),
        horizontes=[f"H{i+1}" for i in range(int(n_steps_forecast))],
        dias_pronostico=int(n_steps_forecast)
    )


@app.get("/info_modelo/{dias_volatilidad}", response_model=InfoModelo, summary="Información del modelo", tags=["Modelo"])
def info_modelo(dias_volatilidad: int):
    if dias_volatilidad not in modelos:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")

    model_data = modelos[dias_volatilidad]
    parametros = convertir_numpy(model_data.get('parametros', {}))
    metricas = convertir_numpy(model_data.get('metricas', {}))

    return InfoModelo(
        dias_volatilidad=int(dias_volatilidad),
        parametros=parametros,
        metricas=metricas
    )