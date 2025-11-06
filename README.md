# Sistema Automatizado para el Prediagnóstico de Bruxismo mediante Análisis de Imágenes Linguales

Este repositorio contiene el desarrollo de un sistema orientado al prediagnóstico del bruxismo utilizando técnicas de visión por computadora y aprendizaje profundo. El proyecto forma parte de una investigación académica que busca ofrecer una alternativa tecnológica accesible y no invasiva para la detección temprana de esta condición.

---

## Descripción del Proyecto
El bruxismo es una disfunción caracterizada por el apretamiento o rechinamiento involuntario de los dientes, con consecuencias clínicas importantes si no se detecta a tiempo. Los métodos tradicionales, como la polisomnografía, son costosos y poco accesibles en muchos contextos.  
Este trabajo propone un sistema que analiza imágenes linguales para identificar marcas dentales en los bordes laterales, una señal morfológica reconocida en la literatura como indicativa de bruxismo.

---

## Objetivos
- Diseñar un modelo basado en aprendizaje profundo capaz de reconocer patrones visuales asociados al bruxismo.
- Implementar un flujo completo de procesamiento de imágenes, desde la segmentación hasta la clasificación.
- Desarrollar un prototipo web que permita realizar predicciones en tiempo real.

---

## Metodología
- **Segmentación automática**: Uso de modelos avanzados como TongueSAM.
- **Preprocesamiento**: Normalización, aumentos de datos y balanceo de clases.
- **Modelado**: Arquitecturas CNN y Vision Transformers (EfficientNetV2, ConvNeXt, Swin Transformer, ViT).
- **Evaluación**: Métricas como accuracy, sensibilidad, especificidad, F1-score y AUC.
- **Despliegue**: Implementación de una aplicación web funcional.

---

## Estructura del Proyecto
- `app.py` → Aplicación principal (interfaz web).
- `requirements.txt` → Dependencias del proyecto.
- `Dockerfile` → Configuración para despliegue en contenedores.
- `utils.py` → Funciones auxiliares.
- `segment/` y `segment_anything/` → Algoritmos de segmentación.
- `static/` → Recursos estáticos (CSS, JS, imágenes).
- `templates/` → Plantillas HTML para la interfaz.

---

## Nota Importante
Este sistema no reemplaza el diagnóstico médico profesional. Su propósito es servir como herramienta de apoyo clínico para la detección temprana del bruxismo en entornos con recursos limitados.  
El contenido de este repositorio está destinado exclusivamente para fines académicos.

---

## Autor
**Anderson Calloquispe**  
Trabajo desarrollado como parte de la tesis universitaria.

---

## Licencia
Uso restringido para fines académicos.
