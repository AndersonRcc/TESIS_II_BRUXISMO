import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from datetime import datetime
import logging # Importar la librería de logging

# Importaciones de tus utilidades para cargar y procesar modelos
from utils import cargar_modelos, segmentar_y_predecir

app = Flask(__name__)
app.secret_key = 'caguero123'

# --- CONFIGURACIÓN DE LOGGING ---
# Esto asegurará que los logs de tu aplicación aparezcan en el "Log stream" de Azure
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO) # Nivel de log para la aplicación Flask

# Aumentar límite de carga
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 

# Ruta base del proyecto
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Carpetas para imágenes subidas y segmentadas
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Configurar en Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# --- ESTADO DE CARGA DEL MODELO Y CARGA INICIAL ---
# Variables globales para los modelos y un flag de estado
global sam_model, model, segment, device, etiquetas, is_model_loaded
sam_model, model, segment, device = None, None, None, None # Inicializar a None
etiquetas = ['MARKED', 'UNMARKED']
is_model_loaded = False # Flag para indicar si los modelos están listos

app.logger.info("APP.PY: Iniciando la carga de modelos de IA al inicio de la aplicación...")
try:
    # Esta llamada se ejecuta una sola vez cuando Gunicorn carga el app.py
    sam_model, model, segment, device = cargar_modelos()
    is_model_loaded = True # Marcar como True solo si la carga fue exitosa
    app.logger.info("APP.PY: Modelos de IA cargados exitosamente.")
except Exception as e:
    app.logger.error(f"APP.PY: Error CRÍTICO al cargar los modelos de IA: {e}", exc_info=True)
    is_model_loaded = False # Asegurar que el flag sea False si falla
    # Aquí la aplicación seguirá, pero el health check fallará y el procesamiento no se intentará.


# --- ENDPOINT DE HEALTH CHECK ---
@app.route('/health', methods=['GET'])
def health_check():
    # Este endpoint permite a Azure saber si la aplicación está lista
    app.logger.info("APP.PY: Solicitud de Health Check recibida.")
    if is_model_loaded and sam_model is not None: # Verifica el flag y que el objeto del modelo no sea None
        app.logger.info("APP.PY: Health Check OK: Modelos cargados y aplicación lista.")
        return jsonify({"status": "healthy", "message": "Modelos cargados y listos."}), 200
    else:
        app.logger.warning("APP.PY: Health Check FALLIDO: Modelos aún no cargados o hubo un error.")
        return jsonify({"status": "unhealthy", "message": "Modelos aún no cargados o fallaron."}), 503


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    app.logger.info("APP.PY: Inicio de la solicitud /upload.")
    # --- VERIFICACIÓN CRUCIAL: ¿Están los modelos cargados? ---
    if not is_model_loaded:
        app.logger.error("APP.PY: Solicitud /upload rechazada: Los modelos de IA aún no están cargados o fallaron al cargar.")
        return jsonify({"status": "error", "message": "El servicio aún no está disponible. Intente nuevamente en unos instantes."}), 503 # Service Unavailable

    if request.method == 'POST':
        if 'image_file' not in request.files:
            app.logger.error("APP.PY: No se encontró 'image_file' en la solicitud POST /upload.")
            return jsonify({"status": "error", "message": "No se encontró el archivo de imagen."}), 400

        image_file = request.files['image_file']

        if image_file.filename == '':
            app.logger.error("APP.PY: Nombre de archivo vacío en solicitud POST /upload.")
            return jsonify({"status": "error", "message": "No se seleccionó ningún archivo."}), 400

        if image_file:
            try:
                filename = secure_filename(image_file.filename)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                base, ext = os.path.splitext(filename)
                unique_filename = f"{base}_{timestamp}{ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

                app.logger.info(f"APP.PY: Guardando imagen subida: {unique_filename}")
                image_file.save(filepath)
                app.logger.info(f"APP.PY: Imagen {unique_filename} guardada. Iniciando segmentación y predicción llamando a utils.segmentar_y_predecir.")

                pred, probs, img_segmented = segmentar_y_predecir(filepath, sam_model, model, segment, device, etiquetas)
                app.logger.info(f"APP.PY: Segmentación y predicción completadas exitosamente para {unique_filename}.")

                result_filename = f"segmented_{base}_{timestamp}{ext}"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                img_segmented.save(result_path)
                app.logger.info(f"APP.PY: Imagen segmentada guardada: {result_filename}.")

                session['pred'] = etiquetas[pred]
                session['probs'] = probs
                session['result_img'] = result_filename

                app.logger.info(f"APP.PY: Solicitud /upload completada con éxito. Redirigiendo a resultado.")
                return jsonify({
                    "status": "success",
                    "message": "Imagen subida y procesada con éxito.",
                    "redirect_url": url_for('result')
                }), 200

            except Exception as e:
                app.logger.error(f"APP.PY: Error durante el procesamiento de la imagen en /upload: {e}", exc_info=True)
                return jsonify({"status": "error", "message": f"Error al procesar la imagen: {str(e)}"}), 500

    return render_template('upload.html')


@app.route('/capture', methods=['GET', 'POST'])
def capture():
    app.logger.info("APP.PY: Inicio de la solicitud /capture.")
    # --- VERIFICACIÓN CRUCIAL: ¿Están los modelos cargados? ---
    if not is_model_loaded:
        app.logger.error("APP.PY: Solicitud /capture rechazada: Los modelos de IA aún no están cargados o fallaron al cargar.")
        return jsonify({"status": "error", "message": "El servicio aún no está disponible. Intente nuevamente en unos instantes."}), 503 # Service Unavailable

    if request.method == 'POST':
        data = request.get_json()

        if not data:
            app.logger.error("APP.PY: No se recibieron datos JSON en solicitud POST /capture.")
            return jsonify({"status": "error", "message": "No se recibieron datos JSON. La cabecera Content-Type podría ser incorrecta o el cuerpo no es JSON válido."}), 400

        data_url = data.get('image_data')

        if not data_url:
            app.logger.error("APP.PY: No se encontró 'image_data' en los datos JSON en /capture.")
            return jsonify({"status": "error", "message": "No se encontró 'image_data' en los datos JSON."}), 400

        try:
            header, encoded = data_url.split(',', 1)
        except ValueError:
            app.logger.error("APP.PY: Formato de imagen Base64 inválido en /capture.")
            return jsonify({"status": "error", "message": "Formato de imagen Base64 inválido."}), 400

        img_data = base64.b64decode(encoded)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = secure_filename(f'captured_{timestamp}.jpeg')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        app.logger.info(f"APP.PY: Guardando imagen capturada: {filename}")
        with open(filepath, 'wb') as f:
            f.write(img_data)
        app.logger.info(f"APP.PY: Imagen {filename} guardada. Iniciando segmentación y predicción llamando a utils.segmentar_y_predecir.")

        try:
            pred, probs, img_segmented = segmentar_y_predecir(filepath, sam_model, model, segment, device, etiquetas)
            app.logger.info(f"APP.PY: Segmentación y predicción completadas exitosamente para {filename}.")
        except Exception as e:
            app.logger.error(f"APP.PY: Error al procesar la imagen con el modelo en /capture: {e}", exc_info=True)
            return jsonify({"status": "error", "message": f"Error al procesar la imagen: {str(e)}"}), 500

        result_filename = f'segmented_{timestamp}.jpeg'
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        img_segmented.save(result_path)
        app.logger.info(f"APP.PY: Imagen segmentada guardada: {result_filename}.")

        session['pred'] = etiquetas[pred]
        session['probs'] = probs
        session['result_img'] = result_filename

        app.logger.info(f"APP.PY: Solicitud /capture completada con éxito. Redirigiendo a resultado.")
        return jsonify({
            "status": "success",
            "message": "Imagen procesada con éxito.",
            "redirect_url": url_for('result')
        }), 200

    return render_template('capture.html')

@app.route('/result')
def result():
    pred = session.get('pred', None)
    probs = session.get('probs', None)
    result_img = session.get('result_img', None)

    if pred is None or probs is None or result_img is None:
        app.logger.warning("APP.PY: Redirigiendo desde /result a /index: Faltan datos de sesión.")
        return redirect(url_for('index'))

    app.logger.info(f"APP.PY: Mostrando resultados: Predicción={pred}, Imagen={result_img}")
    return render_template('result.html', pred=pred, probs=probs, result_img=result_img)

if __name__ == '__main__':
    # Esto solo se ejecuta si corres el script directamente (ej. python app.py)
    # Gunicorn no usa este bloque, pero es útil para desarrollo local.
    app.logger.info("APP.PY: Aplicación Flask iniciada en modo de desarrollo (solo si se ejecuta directamente).")
    app.run(host="0.0.0.0", port=8000)