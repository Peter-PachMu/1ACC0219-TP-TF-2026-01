<div style="width: 100%; display: flex; align-items: flex-start; justify-content: space-between;">
  <div style="width: 50%;">
    <img src="https://static.wikia.nocookie.net/logopedia/images/2/2d/UPC-Logo-Actual.png/revision/latest?cb=20230305155749&path-prefix=es" style="width: 300px; height: auto;">
  </div>
  <div style="width: 50%; text-align: left;">
    <p style="margin: 0; padding-top: 22px;">1ACC0219-2610-3233 - Aplicaciones de Data Science</p>
    <p style="margin: 0;">2026 · Ciencia de la Computación</p>
    <p style="margin: 0;">Profesor: <b>Carlos Fernando Montoya Cubas</b></p>
    <p style="margin: 0;"><b>Alumnos:</b></p>
    <p style="margin: 0;"><b>Avalos Sánchez, César Gabriel</b></p>
    <p style="margin: 0;"><b>Luna Centeno, Sebastian Rodrigo</b></p>
    <p style="margin: 0;"><b>Pacherres Muñoz, Peter Smith</b></p>
  </div>
</div>

---

# Monitoreo Inteligente de Tráfico Vehicular mediante Visión por Computadora

Este proyecto aborda la detección y clasificación de categorías vehiculares utilizando arquitecturas avanzadas de aprendizaje profundo (Deep Learning). A través del uso de modelos convolucionales como **YOLOv8** y arquitecturas basadas en atención como **Vision Transformers (ViT)**, se automatiza la generación de metadatos operativos para el diagnóstico vial y la optimización del flujo en entornos urbanos reales.

## • Objetivo
El objetivo principal del proyecto es desarrollar e implementar un sistema automatizado capaz de identificar y clasificar con precisión cinco categorías principales de vehículos (Ambulancias, Autobuses, Autos, Motocicletas y Camiones) en condiciones urbanas complejas y variables. 

De manera específica, el modelo busca dar respuesta a las siguientes preguntas de investigación:
1. **Clasificación de Categoría de Vehículo:** Determinar la precisión con la que una red neuronal puede categorizar el parque automotor.
2. **Identificación de Unidades de Emergencia:** Evaluar la capacidad del sistema para distinguir prioritariamente vehículos de respuesta a emergencias (Ambulancias) frente al transporte civil o comercial.
3. **Evaluación de Densidad de Carga Pesada:** Medir la eficiencia en la predicción y detección de vehículos de carga y transporte masivo (Trucks y Buses) frente a vehículos ligeros para mejorar la gestión vial.

## • Dataset
El proyecto utiliza el conjunto de datos **"Vehicles OpenImages"** extraído de Roboflow (un subgrupo del ecosistema *Google Open Images*). Las imágenes se encuentran en formato `.jpg` en espacio de color RGB y cuentan con anotaciones de cajas delimitadoras (Bounding Boxes) en formato YOLO (`.txt`).

### Distribución del Dataset en el Experimento:
* **Set de Entrenamiento (Train):** 878 imágenes (Ambulance: 170, Bus: 198, Car: 914, Motorcycle: 202, Truck: 192)
* **Set de Validación (Validation):** 250 imágenes (Ambulance: 64, Bus: 46, Car: 238, Motorcycle: 46, Truck: 60)
* **Set de Prueba (Test):** 126 imágenes (Ambulance: 18, Bus: 38, Car: 150, Motorcycle: 32, Truck: 20)

## • Conclusiones

1. **Eficiencia en Tiempo Real con YOLOv8:** El modelo YOLOv8n demostró ser una solución robusta y ágil para la detección y localización simultánea en video. Consiguió una Precisión general de **0.6706**, un Recall de **0.5524** y un **mAP50 de 0.5837**, destacando especialmente en la clase *Ambulance* con un mAP50 de **0.865** e inferencias promedio de apenas 2.8 ms por imagen.
2. **Importancia del Preprocesamiento en ViT:** En las pruebas iniciales con Vision Transformer (`google/vit-base-patch16-224`) sobre la imagen completa, se identificó un rendimiento deficiente (Accuracy de ~0.25 - 0.37). El análisis de mapas de auto-atención reveló que el token `[CLS]` se distraía con elementos irrelevantes del entorno como postes, fachadas y cables eléctricos.
3. **Impacto de la Optimización (ViT V2):** Al implementar el recorte enfocado de objetos (*bounding box crop* con 15% de margen de contexto), congelar las primeras 8 capas del encoder, aplicar suavizado de pesos de clase (`sqrt(total/count)`) y usar *label smoothing*, el Vision Transformer mejoró radicalmente alcanzando un **Accuracy global del 93%** en el set de prueba.
4. **Métricas de Clasificación por Clase (ViT Optimizado):**
   * **Ambulance:** Precision: 1.00 | Recall: 1.00 | F1-Score: 1.00
   * **Bus:** Precision: 0.94 | Recall: 0.89 | F1-Score: 0.92
   * **Car:** Precision: 0.95 | Recall: 0.91 | F1-Score: 0.93
   * **Motorcycle:** Precision: 0.88 | Recall: 0.94 | F1-Score: 0.91
   * **Truck:** Precision: 0.77 | Recall: 1.00 | F1-Score: 0.87
5. **Validación de Explicabilidad:** Tanto la técnica de perturbación por superpíxeles (**SLIC**) en YOLOv8 como los mapas de calor de la última capa de atención en el ViT V2 comprobaron visualmente que los modelos finales concentran sus pesos matemáticos sobre los contornos intrínsecos de los vehículos, disminuyendo los sesgos del fondo urbano.

## • Licencia
Este proyecto se distribuye bajo la licencia **MIT**. Eres libre de utilizar, modificar y distribuir este software con fines académicos o comerciales siempre que se otorgue el crédito correspondiente a los autores originales.
