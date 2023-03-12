from typing import Dict, Any
import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import get_params, save_artifact


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.params: Dict[str, Any] = get_params(r"./conf/parameters.yml")

    def initiate_ingestion(self):
        try:
            logging.info("Data ingestion initiated")
            df = pd.read_csv(r"./notebooks/data/student.csv")
            save_artifact(self.ingestion_config.raw_data_path, df)

            logging.info("Splitting data into train and test sets")
            df_train, df_test = train_test_split(
                df, 
                test_size=self.params["test_size"], 
                random_state=self.params["random_state"]
            )
            save_artifact(self.ingestion_config.train_data_path, df_train)
            save_artifact(self.ingestion_config.test_data_path, df_test)
            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            ) 
        except Exception as err:
            raise CustomException(err, sys)
        
if __name__=="__main__":
    ingest = DataIngestion()
    ingest.initiate_ingestion()
