from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import uuid, time, base64, os
from functools import lru_cache
from typing import List
import numpy as np

app = FastAPI()

MODEL_DIR = "models/"
DEFAULT_MODEL = "yolov8n.pt"
MAX_IMAGE_SIZE_MB = 5

@lru_cache(maxsize=3)
def load_model(model_name: str = DEFAULT_MODEL):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} not found 💥")
    return YOLO(model_path)

def validate_image(file: UploadFile):
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type 🛑")
    if file.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(status_code=400, detail="Unsupported MIME type 😤")

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = DEFAULT_MODEL
):
    validate_image(file)
    image_bytes = await file.read()

    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"Image too large! ({size_mb:.2f} MB) 😿")

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        model = load_model(model_name)
        start = time.perf_counter()
        results = model(img)
        duration_ms = (time.perf_counter() - start) * 1000

        predictions = results[0].boxes.data.tolist()
        names = results[0].names

        # 🖼️ Create thumbnail preview
        thumbnail_io = BytesIO()
        img.thumbnail((128, 128))
        img.save(thumbnail_io, format="JPEG")
        base64_preview = base64.b64encode(thumbnail_io.getvalue()).decode()

        detections = [
            {
                "label": names[int(pred[5])],
                "confidence": float(pred[4]),
                "bbox": [round(float(x), 2) for x in pred[:4]]
            }
            for pred in predictions
        ]

        return JSONResponse({
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "detections": detections,
            "inference_time_ms": round(duration_ms, 2),
            "thumbnail_base64": base64_preview
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
