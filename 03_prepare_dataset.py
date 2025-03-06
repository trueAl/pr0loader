import os
import sys
import logging
from typing import Any, Tuple
import dotenv
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import torch

REQUIRED_CONFIG_KEYS = ['FILESYSTEM_PREFIX']

logging.basicConfig(
    level=logging.INFO,
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
    def is_valid_image(self, filename: str) -> bool:
        """
        Check if the given filename corresponds to a valid image file.

        Args:
            filename (str): The name of the image file to check.
            img_prefix (str): The directory path prefix where the image file is located.

        Returns:
            bool: True if the file is a valid image, False otherwise.

        Raises:
            Warning: Prints a warning message if the image file is not identified 
            or if there is an error opening the file.
        """
        full_path = os.path.join(self.img_prefix, filename)
        logging.debug("Checking image file %s", full_path)
        if not os.path.isfile(full_path):
            return False
        try:
            # Use 'verify' to quickly check if PIL can recognize the image format.
            with Image.open(full_path) as img:
                logging.debug("Verified image file %s", full_path)
                img.verify()
            return True
        except UnidentifiedImageError as e:
            logging.warning("Warning: Unidentified image file %s: %s", full_path, e)
            return False
        except Exception as e:
            logging.warning("Warning: Error opening image file %s: %s", full_path, e)
            return False

    def __init__(self, csv_file: str, img_prefix: str, transform: transforms.Compose) -> None:
        logging.info("Loading dataset from %s", csv_file)
        self.data = pd.read_csv(csv_file)
        self.img_prefix = img_prefix
        self.transform = transform
        logging.info("Dataset loaded with %d samples", len(self.data))
        # removed the cleaning, instead moved to script 02_clean_dataset.py
        # to clean up the dataset before loading it here, once
        # logging.info("Filtering out invalid image files")
        # original_count = len(self.data)
        # self.data = self.data[self.data['image'].apply(self.is_valid_image)]
        # filtered_count = len(self.data)
        # if filtered_count < original_count:
        #    logging.info("Filtered out %d entries due to invalid or missing image files.",
        #                 original_count - filtered_count)

    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Tuple[Any, dict]:
        row =  self.data.iloc[idx]
        img_path = os.path.join(self.img_prefix, row['image'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, row.to_dict()

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

def process_dataset(my_dataset):
    """
    Processes a dataset by extracting images and metadata, batching the images into a tensor, 
    and saving the processed data to a file.
    Args:
        my_dataset (iterable): An iterable where each item is a tuple containing
        an image and its metadata.
    Returns:
        None
    Side Effects:
        - Logs the progress of processing each sample.
        - Saves the processed data (images and metadata) to 'processed_data.pt'.
    Example:
        >>> my_dataset = [(image1, metadata1), (image2, metadata2)]
        >>> process_dataset(my_dataset)
    """
    all_images = []
    all_metadata = []
    dataset_len = len(my_dataset)

    for idx, (image, metadata) in enumerate(my_dataset):
        all_images.append(image)
        all_metadata.append(metadata)
        logging.info("Processed sample %d/%d", idx + 1, dataset_len)
    logging.info("Finished processing the dataset")
    batched_image_tensor = torch.stack(all_images)

    processed_data = {
        'images': batched_image_tensor,
        'metadata': all_metadata
    }
    torch.save(processed_data, 'processed_data.pt')
    logging.info("Processed data saved successfully to processed_data.pt")

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

    # process_dataset(my_dataset)

if __name__ == "__main__":
    main()
