# Usa la imagen base de Python 3.10
FROM python:3.10-slim-buster

# Instala Git y otras dependencias del sistema operativo (build-essential, librerías gráficas)
# 'git' es crucial para instalar segment-anything desde GitHub.
# 'build-essential' es para compilar dependencias si es necesario (ej. algunas partes de OpenCV/Scikit-image).
# 'libgl1-mesa-glx' y 'libglib2.0-0' son dependencias comunes para librerías de imagen.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo estándar para Azure App Service
WORKDIR /home/site/wwwroot

# Copia los requirements.txt primero para aprovechar el caché de Docker
COPY requirements.txt .

# Instala las dependencias de Python
# '--no-cache-dir' para reducir el tamaño de la imagen final.
# '--find-links' es opcional pero útil si PyTorch/TorchVision no se encuentran por defecto en pip.
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu aplicación al directorio de trabajo
COPY . .

# Expone el puerto en el que Gunicorn escuchará
EXPOSE 8000

# Comando para iniciar la aplicación con Gunicorn
# 'app:app' indica que tu objeto Flask se llama 'app' y está en el módulo 'app.py'.
# '--bind=0.0.0.0:8000' para escuchar en todas las interfaces en el puerto 8000.
# '--timeout 300' para dar 5 minutos al inicio de la app.
CMD ["gunicorn", "app:app", "--bind=0.0.0.0:8000", "--timeout", "300"]