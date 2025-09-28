# Imagen base con Python
FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements de la API
COPY requirements_api.txt .

# Instalar dependencias de la API
RUN pip install --no-cache-dir -r requirements_api.txt

# Copiar la app
COPY app/ ./app

# Exponer puerto
EXPOSE 8000

# Comando para levantar la API con Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

