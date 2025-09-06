import os
import pandas as pd
from src.logger import get_logger
from src.exception import CustomException
import yaml
import pandas as pd

logger = get_logger(__name__)

def read_yaml(file_path: str) -> dict:
    """Reads a YAML file and returns its contents as a dictionary."""
    try:
        if not os.path.exists(file_path):
            raise CustomException(f"YAML file {file_path} does not exist.")
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            logger.info(f"YAML file {file_path} read successfully.")
            return content
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {e}")
        raise CustomException("failed to load the yaml file", e)
    
def load_data(path):
    try:
        logger.info("Loading data...")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise CustomException("failed to load the data", e)
