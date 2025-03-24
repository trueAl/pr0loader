import os
import sys
import logging
import json
import datetime
from typing import Any, Tuple
import dotenv
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import h5py

REQUIRED_CONFIG_KEYS = ['FILESYSTEM_PREFIX']

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] --- %(levelname)s --- %(message)s',
    datefmt='%m/%d/%Y, %H:%M:%S'
)


def validate_config(config: dict) -> bool:
    """
    Validates the given configuration dictionary by checking for the presence of required keys.

    Args:
        config (dict): The configuration dictionary to validate.

    Returns:
        bool: True if all required keys are present, False otherwise. 
        Logs an error if any required keys are missing.
    """
    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        logging.error("Missing required configuration keys: %s", ', '.join(missing_keys))
        return False
    return True

class ImageDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding metadata from a CSV file.

    Attributes:
        data (pd.DataFrame): DataFrame containing the dataset information loaded from the CSV file.
        img_prefix (str): Prefix path to the directory containing the images.
        transform (transforms.Compose): Transformations to be applied to the images.

    Methods:
        __len__() -> int:
            Returns the number of samples in the dataset.
        
        __getitem__(idx: int) -> Tuple[Any, dict]:
            Retrieves the image and its corresponding metadata at the specified index.
            Args:
                idx (int): Index of the sample to retrieve.
            Returns:
                Tuple containing the transformed image and a dictionary of the metadata.
    """

    def __init__(self, csv_file: str, img_prefix: str, transform: transforms.Compose) -> None:
        logging.info("Loading dataset from %s", csv_file)
        self.data = pd.read_csv(csv_file)
        self.img_prefix = img_prefix
        self.transform = transform
        logging.info("Dataset loaded with %d samples", len(self.data))

    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Tuple[Any, dict]:
        print(self.data.iloc[idx, 1])
        img_path = os.path.join(self.img_prefix, self.data.iloc[idx, 1])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        labels = json.dumps(self.data.iloc[idx, 2:].to_dict())
        print(labels)
        return img, labels

def load_image_dataset(config):
    """
    Loads an image dataset using the provided configuration.
    Args:
        config (dict): A dictionary containing configuration parameters. 
                       Expected key is 'FILESYSTEM_PREFIX' which is used as the 
                       prefix for image paths.
    Returns:
        ImageDataset: An instance of the ImageDataset class with the specified 
        transformations applied.
    """
    my_dataset = ImageDataset(
        csv_file="input.csv",
        img_prefix=config['FILESYSTEM_PREFIX'],
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    return my_dataset

def generate_timestamped_filename(prefix: str = 'output', extension: str = 'csv') -> str:
    """
    Generate a timestamped filename with the given prefix and extension.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{prefix}_{timestamp}.{extension}'

def create_hdf5_dataset(my_dataset):
    """
    Creates an HDF5 dataset by extracting images and metadata, and saving them to an HDF5 file.
    Args:
        my_dataset (iterable): An iterable where each item is a tuple containing
        an image and its metadata.
    Returns:
        None
    Side Effects:
        - Logs the progress of processing each sample.
        - Saves the processed data (images and metadata) to 'processed_data.h5'.
    Example:
        >>> my_dataset = [(image1, metadata1), (image2, metadata2)]
        >>> create_hdf5_dataset(my_dataset)
    """
    dataset_len = len(my_dataset)

    batch_size = 32
    num_workers = 16
    data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=False)
    filename=generate_timestamped_filename(prefix='processed_data', extension='h5')
    file_path = os.path.join("Z:\\", filename)

    with h5py.File(file_path, 'w') as f:
        images = f.create_dataset('images', (dataset_len, 3, 224, 224), dtype='f')
        metadata = f.create_dataset('metadata', (dataset_len,), dtype=h5py.special_dtype(vlen=str))
        current_index = 0
        for batch_images, batch_metadata in data_loader:
            batch_size = batch_images.size(0)
            end_index = current_index + batch_size
            images[current_index:end_index] = batch_images
            metadata[current_index:end_index] = batch_metadata
            current_index = end_index
            logging.info("Processed %d/%d samples", current_index, dataset_len)
        logging.info("Finished processing the dataset")

def main() -> None:
    """
    Main function to prepare the dataset
    """
    config = {
        **dotenv.dotenv_values(".env"),
        **os.environ
    }
    # Validate configuration
    if not validate_config(config):
        sys.exit(1)

    my_dataset = load_image_dataset(config)

    logging.info("Dataset loaded successfully with %d samples", len(my_dataset))
    logging.info("Starting with processing of the dataset")

    create_hdf5_dataset(my_dataset)

if __name__ == "__main__":
    main()
