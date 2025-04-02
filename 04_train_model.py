import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import io
import pyarrow.parquet as pq
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# Enable DEV_MODE to run a quick test on a small dataset slice
DEV_MODE = True
# Path to your parquet dataset
PARQUET_PATH = "Z:/processed_data_latest.parquet"

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] --- %(levelname)s --- %(message)s',
    datefmt='%m/%d/%Y, %H:%M:%S'
)


def jpeg_bytes_to_tensor(jpeg_bytes):
    """
    Converts JPEG byte data to a normalized image tensor.
    """
    img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)


class ParquetImageDataset(Dataset):
    """
    Dataset class that reads images and multi-labels from a Parquet file.
    The labels are collected from multiple tag columns and encoded using MultiLabelBinarizer.
    """
    def __init__(self, parquet_file):
        logging.info("Loading Parquet file: %s", parquet_file)
        table = pq.read_table(parquet_file)
        self.df = table.to_pandas()
        if DEV_MODE:
            self.df = self.df.head(10)
            logging.debug("DEV_MODE active: using only first 10 samples")

        self.tag_columns = ['tag1', 'tag2', 'tag3', 'tag4', 'tag5']

        # Combine tags into a list of lists and fit MultiLabelBinarizer
        tag_lists = self.df[self.tag_columns].values.tolist()
        self.mlb = MultiLabelBinarizer()
        self.multi_hot_labels = self.mlb.fit_transform(tag_lists)
        self.label_classes = self.mlb.classes_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_tensor = jpeg_bytes_to_tensor(self.df.iloc[idx]['image'])
        labels = torch.tensor(self.multi_hot_labels[idx], dtype=torch.float32)
        return img_tensor, labels


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set using F1 score and classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().int().tolist())
            all_labels.extend(labels.cpu().int().tolist())

    f1 = f1_score(all_labels, all_preds, average='micro')
    report = classification_report(all_labels, all_preds, digits=4)
    logging.info(f"Validation micro-F1: {f1:.4f}")
    logging.info("Classification Report:\n" + report)


def train():
    """
    Main training loop for multi-label classification with dynamic tag vocabulary.
    Includes:
    - Dataset preparation
    - Model initialization
    - Mixed precision training
    - Learning rate scheduling
    - Evaluation and checkpoint saving
    """
    # Load dataset and split into training and validation sets
    full_dataset = ParquetImageDataset(PARQUET_PATH)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Use high-performance dataloading with CPU resources
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=os.cpu_count() // 2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                            num_workers=os.cpu_count() // 4, pin_memory=True, persistent_workers=True)

    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load ResNet18 and adapt final layer to output as many nodes as tags
    num_classes = len(full_dataset.label_classes)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Use binary cross-entropy for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(25):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if i % 1 == 0:
                logging.info(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")

        scheduler.step()
        logging.info(f"Epoch {epoch+1} complete. Running Loss: {running_loss:.4f}")
        evaluate(model, val_loader, device)

    # Save the final trained model and tag vocabulary
    model_path = "trained_resnet18_multilabel.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'tag_classes': full_dataset.label_classes.tolist()
    }, model_path)
    logging.info(f"Model saved to {model_path}")


if __name__ == '__main__':
    train()
