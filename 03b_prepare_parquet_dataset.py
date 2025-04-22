import os
import sys
from io import BytesIO
import logging
import json
import datetime
from typing import Any, Tuple
import dotenv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REQUIRED_CONFIG_KEYS = ['FILESYSTEM_PREFIX']
DEV_MODE = False  # Set to True to only process the first 10 lines of the CSV

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] --- %(levelname)s --- %(message)s',
    datefmt='%m/%d/%Y, %H:%M:%S'
)


def validate_config(config: dict) -> bool:
    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        logging.error("Missing required configuration keys: %s", ', '.join(missing_keys))
        return False
    return True


class ImageDataset(Dataset):
    def __init__(self, csv_file: str, img_prefix: str, transform: transforms.Compose) -> None:
        logging.info("Loading dataset from %s", csv_file)
        self.data = pd.read_csv(csv_file,
                                keep_default_na=False,
                                dtype=str,                   # Make sure all tags are strings
                                skipinitialspace=True,       # If there are accidental spaces after commas
                                encoding='utf-8')            # Or change based on your CSV source)
        if DEV_MODE:
            self.data = self.data.head(10)
            logging.debug("DEV_MODE active: using only first 10 rows of dataset")
        self.img_prefix = img_prefix
        self.transform = transform
        logging.info("Dataset loaded with %d samples", len(self.data))

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, dict]:
        img_path = os.path.join(self.img_prefix, self.data.iloc[idx, 1])
        metadata = self.data.iloc[idx, 2:].to_dict()
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error("There was an error dealing with %s: %s", img_path, e)
            return None
        img = self.transform(img)
        logging.debug(
            "Loaded image '%s' | shape: %s | type: %s (%s) | metadata: %s",
            img_path,
            img.shape,
            type(img).__name__,
            img.__class__,
            ", ".join(f"{k}={v}" for k, v in metadata.items())
        )
        return img, metadata

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # If the batch is empty, return None
    if len(batch) == 0:
        return None
    # Otherwise, use the default_collate function
    return torch.utils.data.dataloader.default_collate(batch)


def load_image_dataset(config):
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


def generate_timestamped_filename(prefix: str = 'output', extension: str = 'parquet') -> str:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{prefix}_{timestamp}.{extension}'


def image_tensor_to_jpeg_bytes(image_tensor):
    image_array = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image_array = ((image_array * 255).clip(0, 255)).astype('uint8')
    img = Image.fromarray(image_array)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def create_parquet_dataset(my_dataset):
    dataset_len = len(my_dataset)
    batch_size = 1
    num_workers = 16
    data_loader = DataLoader(my_dataset, 
                             batch_size=batch_size,
                             num_workers=num_workers, 
                             shuffle=False,
                             collate_fn=collate_fn)

    filename = generate_timestamped_filename(prefix='processed_data', extension='parquet')
    file_path = os.path.join("Y:\\", filename)

    parquet_writer = None
    current_index = 0
    batch_number = 0  # Introduce a batch counter
    try:
        for batch in data_loader:
            if batch is None:
                continue  # Skip empty batches
            batch_images, batch_metadata = batch
            batch_number += 1
            image_bytes_list = [image_tensor_to_jpeg_bytes(img) for img in batch_images]

            df_metadata = pd.DataFrame(batch_metadata)
            df_metadata.insert(0, "image", image_bytes_list)

            table = pa.Table.from_pandas(df_metadata)

            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(file_path, table.schema, compression='snappy')

            parquet_writer.write_table(table)

            current_index += batch_images.size(0)
            logging.info("Processed %d/%d samples", current_index, dataset_len)
            
    except Exception as e:
        logging.error(
            "Error at batch #%d | Batch images shape: %s | Metadata keys: %s | Exception: %s",
            batch_number,
            tuple(batch_images.shape) if 'batch_images' in locals() else "unknown",
            list(batch_metadata.keys()) if 'batch_metadata' in locals() and isinstance(batch_metadata, dict) else "unknown",
            str(e)
        )

    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    logging.info("Finished processing the dataset, saved to %s", file_path)

def main() -> None:
    config = {
        **dotenv.dotenv_values(".env"),
        **os.environ
    }
    if not validate_config(config):
        sys.exit(1)

    my_dataset = load_image_dataset(config)
    logging.info("Dataset loaded successfully with %d samples", len(my_dataset))
    logging.info("Starting processing of the dataset")

    create_parquet_dataset(my_dataset)


if __name__ == "__main__":
    main()
