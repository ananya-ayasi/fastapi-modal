# main.py
import torch
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import modal
import numpy as np

# ---- Modal Setup ----
modal_app = modal.App("resnet-inference-app")
image = modal.Image.debian_slim().pip_install("fastapi", "torch", "torchvision", "uvicorn", "pillow", "numpy","requests","python-multipart")

# ---- Model & Preprocessing ----
#model = models.resnet18(pretrained=True)
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# ---- Load class labels ----
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = []
with open("imagenet_classes.txt", "w") as f:
    import requests
    f.write(requests.get(LABELS_URL).text)
with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# ---- FastAPI App ----
fastapi_app = FastAPI()

@fastapi_app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        predicted_label = imagenet_classes[predicted_idx]

    return {"prediction": predicted_label}


# ---- Modal Entrypoint ----
@modal_app.function(image=image, min_containers=1, timeout=60)
@modal.asgi_app()
def fastapi_app():
    return fastapi_app
app= modal_app
