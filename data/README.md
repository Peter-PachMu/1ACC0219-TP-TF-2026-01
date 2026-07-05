# Carpeta de Datos (Data Directory)

Los datos originales utilizados en este proyecto no se almacenan directamente en este repositorio debido a restricciones de espacio y para garantizar la reproducibilidad mediante código. 

Los datos son descargados e introducidos automáticamente durante la ejecución del entorno de Google Colab (`Trabajo_Final_2026.ipynb`).

### Origen del Dataset
* **Nombre:** Vehicles OpenImages
* **Plataforma/Fuente:** Roboflow (Ecosistema Google Open Images)
* **Formato original:** Imágenes `.jpg` (RGB) y anotaciones de bounding boxes en formato YOLO (`.txt`).

### Distribución de los Datos
Si deseas reproducir el entorno de forma local, el dataset está estructurado internamente en las siguientes proporciones:
* **Train:** 878 imágenes (Clases: Ambulance, Bus, Car, Motorcycle, Truck)
* **Validation:** 250 imágenes
* **Test:** 126 imágenes

### Instrucciones de Descarga Automática
Para obtener exactamente los mismos archivos, ejecuta las primeras celdas de preparación de datos en el Notebook de Jupyter, las cuales conectan directamente con la API de Roboflow o el enlace de descarga del dataset precargado.