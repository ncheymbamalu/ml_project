from typing import Dict, Any
import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, 
    AdaBoostRegressor, 
    GradientBoostingRegressor
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import get_params, evaluate_models, save_artifact
from src.components.ingestion import DataIngestion
from src.components.transformation import DataTransformation


@dataclass
class TrainConfig:
    model_path = os.path.join("artifacts", "model.pkl")


class Train:
    def __init__(self):
        self.train_config = TrainConfig()
        self.params: Dict[str, Any] = get_params(r"./conf/parameters.yml")

    def initiate_train(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        try:
            target = self.params["target"]
            X_train, y_train = train_set.drop(target, axis=1), train_set[target]
            X_test, y_test = test_set.drop(target, axis=1), test_set[target]

            # base models
            models = {
                "LinearRegression": LinearRegression(), 
                "RandomForestRegression": RandomForestRegressor(), 
                "AdaBoostRegression": AdaBoostRegressor(), 
                "GradientBoostingRegression": GradientBoostingRegressor(), 
                "CatBoostRegression": CatBoostRegressor(verbose=False), 
                "XGBRegression": XGBRegressor() 
            }

            name, model, score = evaluate_models(
                models, 
                X_train, 
                y_train, 
                X_test, 
                y_test, 
                self.params
            )

            if score < 0.65:
                raise CustomException("No best model found")
            
            logging.info(
                "%s, the highest test set adjusted R², was produced via %s",  
                np.round(score, 2), 
                name
            )

            logging.info(
                "Saving the %s model as './%s'", 
                name,
                self.train_config.model_path
            )
            save_artifact(
                self.train_config.model_path, 
                model
            )

            print(f"{score:.2f}, the highest test set adjusted R², was produced via {name}")
        except Exception as err:
            raise CustomException(err, sys)
        

if __name__=="__main__":
    train_data_path, test_data_path = DataIngestion().initiate_ingestion()
    df_train, df_test, _ = DataTransformation().initiate_transformation(train_data_path, test_data_path)
    Train().initiate_train(df_train, df_test)
