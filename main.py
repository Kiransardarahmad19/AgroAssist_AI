import io
import os
import json
from typing import List, Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tvm

import pickle


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths (customize if needed)
CROP_MODEL_PATH = os.getenv("CROP_MODEL_PATH", "model.pkl")
CUSTOM_CNN_CKPT = os.getenv("CUSTOM_CNN_CKPT", None)       
VGG16_CKPT      = os.getenv("VGG16_CKPT", None)            
RESNET34_CKPT   = os.getenv("RESNET34_CKPT", "resnet34_finetuned.pth")  


DISEASE_CLASSES: List[str] = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


class ImageClassificationBase(nn.Module):
    def training_step(self, batch): ...
    def validation_step(self, batch): ...
    def validation_epoch_end(self, outputs): ...
    def epoch_end(self, epoch, result): ...

class Plant_Disease_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2), # 128x32x32
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2), # 256x16x16
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2), # 512x8x8
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2), # 1024x4x4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024,512), nn.ReLU(),
            nn.Linear(512,256),  nn.ReLU(),
            nn.Linear(256,38)
        )

    def forward(self, x): return self.network(x)

# -- VGG16 (transfer learning) --
class Plant_Disease_Model1(nn.Module):
    def __init__(self):
        super().__init__()
        m = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        num_ftrs = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(num_ftrs, 38)
        self.network = m
    def forward(self, x): return self.network(x)


class Plant_Disease_Model2(nn.Module):
    def __init__(self):
        super().__init__()
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
        num_ftrs = m.fc.in_features
        m.fc = nn.Linear(num_ftrs, 38)
        self.network = m
    def forward(self, x): return self.network(x)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform_224 = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Your custom CNN was (likely) trained on 128x128
transform_128 = T.Compose([
    T.Resize((128,128)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


_loaded_models = {}

def load_disease_model(kind: str) -> nn.Module:
    key = kind.lower()
    if key in _loaded_models:
        return _loaded_models[key]

    if key == "custom":
        model = Plant_Disease_Model()
        ckpt = CUSTOM_CNN_CKPT
    elif key == "vgg16":
        model = Plant_Disease_Model1()
        ckpt = VGG16_CKPT
    elif key == "resnet34":
        model = Plant_Disease_Model2()
        ckpt = RESNET34_CKPT
    else:
        raise HTTPException(status_code=400, detail="model must be one of: custom, vgg16, resnet34")


    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    model.eval().to(DEVICE)
    _loaded_models[key] = model
    return model

def preprocess_image(file_bytes: bytes, size: int) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    x = transform_224(img) if size == 224 else transform_128(img)
    return x.unsqueeze(0).to(DEVICE)

def softmax_np(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
    return probs


_crop_model = None
def load_crop_model():
    global _crop_model
    if _crop_model is not None:
        return _crop_model
    if not os.path.exists(CROP_MODEL_PATH):
        raise RuntimeError(f"Crop model file not found: {CROP_MODEL_PATH}")
    with open(CROP_MODEL_PATH, "rb") as f:
        _crop_model = pickle.load(f)
    return _crop_model


app = FastAPI(title="Agri ML Inference API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "models_cached": list(_loaded_models.keys())}

# ---------- Meta (classes) ----------
@app.get("/meta/disease-classes")
def meta_disease_classes():
    return {"count": len(DISEASE_CLASSES), "classes": DISEASE_CLASSES}

@app.get("/meta/crop-classes")
def meta_crop_classes():
    model = load_crop_model()
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        for step in ["gaussiannb","nb","clf","classifier"]:
            if step in model.named_steps and hasattr(model.named_steps[step], "classes_"):
                classes = model.named_steps[step].classes_
                break
    if classes is None:
        raise HTTPException(status_code=500, detail="Could not find classes_ in the crop model.")
    return {"count": int(len(classes)), "classes": [str(c) for c in classes]}


class DiseaseResponse(BaseModel):
    model: str
    topk: int = 3
    prediction: str
    topk_labels: List[str]
    topk_probs: List[float]

@app.post("/predict/disease", response_model=DiseaseResponse)
async def predict_disease(
    file: UploadFile = File(..., description="Leaf image"),
    model: str = Query("resnet34", enum=["custom", "vgg16", "resnet34"]),
    topk: int = Query(3, ge=1, le=10)
):
    # load and preprocess
    content = await file.read()
    input_size = 224 if model in ("vgg16", "resnet34") else 128
    x = preprocess_image(content, input_size)

    # forward
    net = load_disease_model(model)
    with torch.no_grad():
        logits = net(x)
    probs = softmax_np(logits)

    # top-k
    idx = np.argsort(probs)[::-1][:topk]
    labels = [DISEASE_CLASSES[i] for i in idx]
    p = probs[idx].round(6).tolist()

    return DiseaseResponse(
        model=model,
        topk=topk,
        prediction=labels[0],
        topk_labels=labels,
        topk_probs=p
    )


class CropFeatures(BaseModel):
    N: float = Field(..., description="Nitrogen")
    P: float = Field(..., description="Phosphorus")
    K: float = Field(..., description="Potassium")
    Temperature: float
    Humidity: float
    ph: float
    Rainfall: float

class CropResponse(BaseModel):
    prediction: str
    topk_labels: Optional[List[str]] = None
    topk_probs: Optional[List[float]] = None

@app.post("/predict/crop", response_model=CropResponse)
def predict_crop(features: CropFeatures, topk: int = Query(3, ge=1, le=10)):
    model = load_crop_model()

    row = {
        "N": features.N,
        "P": features.P,
        "K": features.K,
        "temperature": features.Temperature,
        "humidity": features.Humidity,
        "ph": features.ph,
        "rainfall": features.Rainfall,
    }

    try:
        import pandas as pd
        X = pd.DataFrame([row])
        if hasattr(model, "feature_names_in_"):
            X = X.reindex(columns=list(model.feature_names_in_))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pandas error: {e}")

    try:
        y_pred = model.predict(X)[0]
        topk_labels, topk_probs = None, None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            idx = np.argsort(proba)[::-1][:topk]
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                for step in ["gaussiannb","nb","clf","classifier"]:
                    if step in model.named_steps and hasattr(model.named_steps[step], "classes_"):
                        classes = model.named_steps[step].classes_
                        break
            if classes is None:
                classes = np.array([str(i) for i in range(len(proba))])
            topk_labels = [str(classes[i]) for i in idx]
            topk_probs = proba[idx].round(6).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop prediction failed: {e}")

    return CropResponse(prediction=str(y_pred), topk_labels=topk_labels, topk_probs=topk_probs)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
