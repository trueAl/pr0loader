import logging
import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import pyarrow.parquet as pq
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms

# Enable DEV_MODE to run a quick test on a small dataset slice
DEV_MODE = True

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