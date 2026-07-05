"""
Backend Flask para comparar el modelo YOLOv8 (detección) y el modelo ViT
(clasificación) entrenados en el notebook, sobre una misma imagen subida
por el usuario.

Cómo ejecutar:
    1. Coloca tus modelos entrenados dentro de la carpeta `models/` (ver README.md).
    2. pip install -r requirements.txt
    3. python app.py
    4. Abre http://127.0.0.1:5000 en el navegador.
"""

import base64
import io
import os
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, render_template, request

# --------------------------------------------------------------------------
# Configuración de rutas de los modelos 
# de entorno sin tocar el código, por ejemplo:
#   YOLO_MODEL_PATH=/otra/ruta/best.pt python app.py
# --------------------------------------------------------------------------
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", os.path.join("models", "yolo_best.pt"))
VIT_MODEL_PATH = os.environ.get("VIT_MODEL_PATH", os.path.join("models", "vit_vehicle_best_model"))
MAX_UPLOAD_MB = 15

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

device = "cuda" if torch.cuda.is_available() else "cpu"


yolo_model = None
yolo_error = None
try:
    from ultralytics import YOLO

    if os.path.exists(YOLO_MODEL_PATH):
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"[YOLO] Modelo cargado desde: {YOLO_MODEL_PATH}")
    else:
        yolo_error = f"No se encontró el archivo del modelo YOLO en: {YOLO_MODEL_PATH}"
        print(f"[YOLO] AVISO: {yolo_error}")
except Exception as e:  # noqa: BLE001
    yolo_error = f"Error al cargar YOLO: {e}"
    print(f"[YOLO] ERROR: {yolo_error}")

vit_model = None
vit_processor = None
vit_error = None
try:
    from transformers import ViTForImageClassification, ViTImageProcessor

    if os.path.isdir(VIT_MODEL_PATH):
        vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_PATH)
        vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_PATH).to(device)
        vit_model.eval()
        print(f"[ViT] Modelo cargado desde: {VIT_MODEL_PATH}")
    else:
        vit_error = f"No se encontró la carpeta del modelo ViT en: {VIT_MODEL_PATH}"
        print(f"[ViT] AVISO: {vit_error}")
except Exception as e:  # noqa: BLE001
    vit_error = f"Error al cargar ViT: {e}"
    print(f"[ViT] ERROR: {vit_error}")


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def run_yolo(img: Image.Image):
    if yolo_model is None:
        return {"available": False, "error": yolo_error}

    results = yolo_model.predict(img, verbose=False)
    r = results[0]

    # Imagen anotada con las cajas (ultralytics la devuelve en BGR/numpy)
    annotated_bgr = r.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_img = Image.fromarray(annotated_rgb)

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
        detections.append({
            "class": r.names[cls_id],
            "confidence": round(conf, 4),
            "box": xyxy,
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "available": True,
        "annotated_image": pil_to_base64(annotated_img),
        "detections": detections,
        "num_detections": len(detections),
    }


def run_vit(img: Image.Image):
    if vit_model is None:
        return {"available": False, "error": vit_error}

    inputs = vit_processor(images=img.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        logits = vit_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    id2label = vit_model.config.id2label
    ranking = sorted(
        (
            {"class": id2label[i], "confidence": round(float(p), 4)}
            for i, p in enumerate(probs)
        ),
        key=lambda d: d["confidence"],
        reverse=True,
    )

    return {
        "available": True,
        "predictions": ranking,
        "top_class": ranking[0]["class"],
        "top_confidence": ranking[0]["confidence"],
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        yolo_ready=yolo_model is not None,
        vit_ready=vit_model is not None,
        yolo_error=yolo_error,
        vit_error=vit_error,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:  # noqa: BLE001
        return jsonify({"error": "El archivo subido no es una imagen válida."}), 400

    try:
        yolo_result = run_yolo(img)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        yolo_result = {"available": False, "error": f"Error durante la inferencia YOLO: {e}"}

    try:
        vit_result = run_vit(img)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        vit_result = {"available": False, "error": f"Error durante la inferencia ViT: {e}"}

    return jsonify({
        "original_image": pil_to_base64(img),
        "yolo": yolo_result,
        "vit": vit_result,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
