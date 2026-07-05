# Comparador YOLOv8 vs ViT — Vehículos

App local (Flask) para subir una imagen y ver, lado a lado, qué detecta el
modelo **YOLOv8** entrenado y qué clasifica el modelo **ViT** entrenado 

## 1. Estructura esperada

```
vehicle-app/
├── app.py
├── requirements.txt
├── templates/index.html
├── static/...
└── models/
    ├── yolo_best.pt                 <- pesos entrenados de YOLO
    └── vit_vehicle_best_model/      <- carpeta guardada por trainer_v2.save_model()
        ├── config.json
        ├── model.safetensors (o pytorch_model.bin)
        └── preprocessor_config.json
```

### YOLO (`yolo_best.pt`)
```
/content/runs/detect/train-2/weights/best.pt
```

En una celda de Colab, ejecutar para confirmar la ruta exacta y descargarlo:

```python
import glob
from google.colab import files

# Verifica el path real (puede variar el número de la carpeta "train-N")
best_paths = glob.glob('/content/runs/detect/*/weights/best.pt')
print(best_paths)

files.download(best_paths[-1])  # descarga el más reciente
```

Renombrar el archivo descargado a `yolo_best.pt` y colocarlo en `models/`.

### ViT (`vit_vehicle_best_model/`)
El `vit_vehicle_best_model.zip` generado por la celda de "Guardado del
Modelo Entrenado". Descárgarlo y descomprímelo dentro de `models/` de forma
que quede como `models/vit_vehicle_best_model/config.json`, etc.

## 2. Instalación

```bash
cd vehicle-app
python -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Ejecutar

```bash
python app.py
```

Abrir **http://127.0.0.1:5000** en tu navegador.

## 4. Uso

1. Sube o arrastra una imagen con vehículos.
2. Haz clic en **"Analizar con ambos modelos"**.
3. A la izquierda verás las cajas detectadas por YOLO (con su confianza por
   clase); a la derecha, el ranking de probabilidades de ViT para todas las
   clases, con la más alta destacada.

