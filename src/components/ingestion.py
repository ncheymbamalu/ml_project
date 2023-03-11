import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated")
        try:
            logging.info("Dataset read in as a Pandas DataFrame")
            df = pd.read_csv(r"./notebooks/data/student.csv")

            os.makedirs("artifacts", exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Dataset split into train and test sets")
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
            df_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            ) 
        except Exception as err:
            raise CustomException(err, sys)
        
if __name__=="__main__":
    data_ingestion_instance = DataIngestion()
    data_ingestion_instance.initiate_data_ingestion()
