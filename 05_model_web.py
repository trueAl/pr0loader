import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import logging
import pandas as pd
import io

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] --- %(levelname)s --- %(message)s',
    datefmt='%m/%d/%Y, %H:%M:%S'
)

# Image preprocessing transform (same as training)
def preprocess_image(pil_img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0)  # add batch dimension

# Load model and label mapping
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
    tag_classes = checkpoint['tag_classes']

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(tag_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    return model, tag_classes

# Predict tags for a PIL image with confidence scores and export options
def predict_tags_interface(image, threshold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).squeeze().cpu()

    # Prepare DataFrame of tags and confidence scores
    tag_confidences = [(tag_classes[i], float(probs[i])) for i in range(len(tag_classes)) if probs[i] > threshold]
    tag_confidences.sort(key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(tag_confidences, columns=["Tag", "Confidence"])

    # Convert image to preview for output display
    preview_img = image.copy()

    return preview_img, df, df.to_csv(index=False)

# Load model once on startup
model_path = "trained_resnet18_multilabel.pth"
logging.info("Loading model...")
model, tag_classes = load_model(model_path)

# Define Gradio interface
demo = gr.Interface(
    fn=predict_tags_interface,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(0.0, 1.0, value=0.5, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="pil", label="Image Preview"),
        gr.Dataframe(headers=["Tag", "Confidence"], type="pandas", label="Predicted Tags"),
        gr.File(label="Download CSV")
    ],
    title="Image Tag Predictor",
    description="Upload an image and see predicted tags with confidence scores based on ResNet18 multi-label classification. You can adjust the confidence threshold and download results as CSV."
)

if __name__ == "__main__":
    demo.launch()
