from pydantic import BaseModel, Field
from typing import List, Dict


# Entrada de datos
class DatosBTC(BaseModel):
    lags: List[float] = Field(
        ...,
        description="Valores históricos de los lags requeridos por el modelo. La cantidad de lags depende del modelo específico.",
        json_schema_extra=[0.5, 0.6, 0.55, 0.58, 0.57, 0.59, 0.6]
    )
    tipo_volatilidad: int = Field(
        ...,
        description="Tipo de volatilidad a predecir. Debe ser 7, 14, 21 o 28 días",
        json_schema_extra=7
    )


# Respuesta de predicción
class RespuestaPrediccion(BaseModel):
    tipo_volatilidad: int = Field(..., description="Tipo de volatilidad predicha")
    volatilidad_predicha: List[float] = Field(..., description="Los valores de volatilidad predicha para los próximos 7 días")
    horizontes: List[str] = Field(..., description="Nombre de cada horizonte de predicción (H1, H2, …) correspondiente a cada valor predicho")
    dias_pronostico: int = Field(..., description="Cantidad de días en el horizonte de pronóstico (siempre 7)")


# Información del modelo
class InfoModelo(BaseModel):
    dias_volatilidad: int = Field(..., description="Volatilidad que maneja el modelo (en días)")
    parametros: Dict[str, float] = Field(..., description="Parámetros del modelo")
    metricas: Dict[str, float] = Field(..., description="Métricas calculadas del modelo (RMSE, MAE, etc.)")