# Predicción de la Volatilidad de Bitcoin por Ventanas Temporales

Este proyecto implementa un sistema para **predecir la volatilidad del Bitcoin** mediante modelos de redes neuronales **MultiLayer Perceptron (MLP)** entrenados con diferentes ventanas temporales (7, 14, 21 y 28 días).

## Flujo del proyecto

- **Análisis Exploratorio de Datos (EDA)**  
- **Entrenamiento de modelos MLP** para predicción de 7 días futuros  
- **Validación temporal** y evaluación con métricas estadísticas  
- Selección de los **mejores 4 modelos** (uno por cada ventana de volatilidad)  
- Exposición de los modelos en una **API con FastAPI**  
- **Tests unitarios** para validar API y modelos  
- **Contenerización con Docker** para despliegue  

## Objetivos principales

- Implementar una **validación temporal adecuada** para series de tiempo  
- Entrenar y comparar modelos MLP con distintas ventanas de volatilidad  
- Evaluar con métricas como **RMSE, MAE, MAPE, MSE y p-value del test BDS**  
- Servir los modelos entrenados mediante una **API robusta y documentada**  
- Garantizar reproducibilidad mediante **tests y CI/CD con GitHub Actions**

## Tecnologías implementadas

- [Python 3.10+](https://www.python.org/)
- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) → Servir la API  
- [Docker](https://www.docker.com/) → Contenerización  
- [GitHub Actions](https://docs.github.com/en/actions) → CI/CD  
- [scikit-learn](https://scikit-learn.org/stable/) → Modelos MLP (MLPRegressor) y validación temporal (TimeSeriesSplit)  
- [arch](https://arch.readthedocs.io/) → Cálculo de pruebas estadísticas (test BDS)  
- [pandas](https://pandas.pydata.org/) y [numpy](https://numpy.org/) → Manipulación y análisis de datos  
- [matplotlib](https://matplotlib.org/) → Visualización  
- [joblib](https://joblib.readthedocs.io/) → Serialización de modelos  

## Estructura del Proyecto

```
MINIPROYECTO2_VOLATILIDAD/
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI/CD con GitHub Actions
├── app/
│   ├── modelos_finales/              # 4 mejores modelos MLP entrenados (.joblib)
│   ├── api.py                        # API principal con FastAPI
│   └── schemas.py                    # Esquemas Pydantic para la API
├── data/
│   ├── btc_1d_data_2018_to_2025.csv  # Dataset principal de Bitcoin
│   └── datamodelos.csv               # Datos procesados para modelos
├── notebooks/
│   ├── figs/                         # Gráficos y visualizaciones generadas
│   ├── results/                      # Tablas de resultados
│   ├── EDA.ipynb                     # Análisis exploratorio de datos
│   ├── MODELOS.ipynb                 # Entrenamiento y evaluación de modelos
│   └── RESIDUOS.ipynb                # Análisis de residuos y test BDS
├── tests/
│   ├── test_api.py                   # Tests unitarios para la API
│   └── test_model.py                 # Tests para validación de modelos
├── apirequirements.txt               # Dependencias específicas para la API
├── Dockerfile                        # Configuración para contenerización
├── README.md                         # Documentación del proyecto
├── requirements_api.txt              # Dependencias mínimas para producción
└── requirements.txt                  # Todas las dependencias del proyecto
```

## Instrucciones de uso

### Instalación

**Clonar el repositorio**

```bash
git clone <URL_REPOSITORIO>
cd MINIPROYECTO2_VOLATILIDAD
```

**Configurar entorno virtual e instalar dependencias**

```bash
python -m venv btc_env
source btc_env/bin/activate  # Linux/Mac
btc_env\Scripts\activate     # Windows

pip install -r requirements.txt
pip install -r requirements_api.txt
pip install -r apirequirements.txt
```

## Ejecución de la API

### Iniciar servidor de desarrollo

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### Acceso a la API

- **API Principal**: [http://localhost:8000](http://localhost:8000)  
- **Documentación interactiva (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## Contenerización con Docker

### Construir y ejecutar contenedor

```bash
docker build -t btc-volatility-api .
docker run -p 8000:8000 btc-volatility-api
```

Accede a la API en: [http://localhost:8000](http://localhost:8000)

## Ejemplo de uso

### Predicción con modelo de 7 días

```bash
curl -X POST "http://localhost:8000/predecir" \
  -H "Content-Type: application/json" \
  -d '{
    "lags": [0.5, 0.6, 0.55, 0.58, 0.57, 0.59, 0.6],
    "tipo_volatilidad": 7
  }'
```

La API proporciona toda la información necesaria para su correcta implementación, incluyendo cómo y cuántos valores enviar según la ventana de días.

## Ejecución de tests

### Ejecutar todos los tests

```bash
python -m pytest tests/ -v
```

### Ejecutar tests individuales

```bash
python -m pytest tests/test_api.py -v
python -m pytest tests/test_model.py -v
```
