import os
import sys
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.exception import CustomException
from config.path_config import *
from utils.common_function import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        self.train_ratio = self.config["train_ratio"]
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Data Ingestion started with {self.bucket_name} and file is {self.bucket_file_name}")
    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Downloaded {self.bucket_file_name} from GCP bucket {self.bucket_name} to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error downloading file from GCP: {e}")
            raise CustomException("Failed to download the CSV file", sys)
        
    def split_data(self):
        try:
            logger.info("Starting the splitting of data into train and test sets")
            data = pd.read_csv(RAW_FILE_PATH)
            train_data, test_data = train_test_split(data, train_size=self.train_ratio, random_state=42)
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Data split into train and test sets with ratio {self.train_ratio}")
            logger.info(f"Train data saved at {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved at {TEST_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise CustomException("Error during the splitting", sys)  
        
    def run(self):
        try:
            logger.info("Starting the data ingestion process")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion process completed successfully")
        except Exception as e:
            logger.error(f"Error in data ingestion process: {e}")
            raise CustomException("Error in the data ingestion process", sys)
        finally:
            logger.info("Data Ingestion process is completed")

if __name__ == "__main__":
    try:
        config = read_yaml(CONFIG_PATH)
        data_ingestion = DataIngestion(config)
        data_ingestion.run()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        raise CustomException("Error running data ingestion script", sys)
