import torch
from torchvision import models, transforms
from PIL import Image
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import io
import uvicorn

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] --- %(levelname)s --- %(message)s',
    datefmt='%m/%d/%Y, %H:%M:%S'
)

# FastAPI app instance
app = FastAPI(title="Image Tag Prediction API", description="Upload an image and get tag predictions with confidence scores.", version="1.0.0")

# CORS configuration (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model and tags
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
    tag_classes = checkpoint['tag_classes']

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(tag_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    return model, tag_classes

model_path = "trained_resnet18_multilabel.pth"
logging.info("Loading model...")
model, tag_classes = load_model(model_path)

def predict_tags(image_bytes, threshold=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu()

    predictions = [
        {"tag": tag_classes[i], "confidence": float(probs[i])}
        for i in range(len(tag_classes)) if probs[i] > threshold
    ]
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return predictions

@app.post("/predict", summary="Predict Tags", tags=["Prediction"])
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    try:
        image_bytes = await file.read()
        result = predict_tags(image_bytes, threshold)
        return JSONResponse(content={"predictions": result})
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image.")

# Custom OpenAPI schema with security best practices
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run("tag_prediction_api:app", host="0.0.0.0", port=8000, reload=True)
