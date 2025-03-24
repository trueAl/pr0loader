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

REQUIRED_CONFIG_KEYS = ['FILESYSTEM_PREFIX']

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
        self.data = pd.read_csv(csv_file)
        self.img_prefix = img_prefix
        self.transform = transform
        logging.info("Dataset loaded with %d samples", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, dict]:
        img_path = os.path.join(self.img_prefix, self.data.iloc[idx, 1])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        labels = json.dumps(self.data.iloc[idx, 2:].to_dict())
        return img, labels


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
    batch_size = 32
    num_workers = 16
    data_loader = DataLoader(my_dataset, batch_size=batch_size,
                             num_workers=num_workers, shuffle=False)

    filename = generate_timestamped_filename(prefix='processed_data', extension='parquet')
    file_path = os.path.join("Z:\\", filename)

    parquet_writer = None
    current_index = 0

    try:
        for batch_images, batch_metadata in data_loader:
            image_bytes_list = [image_tensor_to_jpeg_bytes(img) for img in batch_images]

            df_batch = pd.DataFrame({
                "image": image_bytes_list,
                "metadata": batch_metadata
            })

            table = pa.Table.from_pandas(df_batch)

            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(file_path, table.schema, compression='snappy')

            parquet_writer.write_table(table)

            current_index += batch_images.size(0)
            logging.info("Processed %d/%d samples", current_index, dataset_len)

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
