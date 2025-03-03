import os
import dotenv
import logging
import pandas as pd
#from torch.utils.data import Dataset, DataLoader


REQUIRED_CONFIG_KEYS = ['FILESYSTEM_PREFIX']

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] --- %(levelname)s --- %(message)s',
    datefmt='%m/%d/%Y, %H:%M:%S'
)


def validate_config(config: dict) -> bool:
    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        logging.error(f"Missing required configuration keys: {', '.join(missing_keys)}")
        return False
    return True

def load_csv(filename) -> pd.DataFrame:
    try:
        df: pd.DataFrame = pd.read_csv(filename)
        return df
    except Exception as e:
            raise RuntimeError(f"Error reading {filename}: {e}")
    
class ImageDataset():
    def __init__() -> None:
          None
    def __len__(self) -> int:
        return len(self.data)
    
    
def main() -> None:
    config = {
        **dotenv.dotenv_values(".env"),
        **os.environ
    }
    # Validate configuration
    if not validate_config(config):
        exit(1)
    myDataset: Dataset = None

    mycsv = load_csv("input.csv")
    print(type(mycsv))
    print(mycsv.head())
    print(mycsv.__len__)


if __name__ == "__main__":
    main()
