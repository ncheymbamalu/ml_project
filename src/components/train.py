from typing import Dict, Any
import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
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

            models = {
                "LinearRegression": LinearRegression(), 
                "LassoRegression": Lasso(), 
                "RidgeRegression": Ridge(), 
                "KNeighborsRegression": KNeighborsRegressor(), 
                "SupportVectorRegression": SVR(), 
                "DecisionTreeRegression": DecisionTreeRegressor(), 
                "RandomForestRegression": RandomForestRegressor(), 
                "AdaBoostRegression": AdaBoostRegressor(), 
                "GradientBoostingRegression": GradientBoostingRegressor(), 
                "CatBoostRegression": CatBoostRegressor(verbose=False), 
                "XGBRegression": XGBRegressor() 
            }

            report = evaluate_models(
                models, 
                X_train, 
                y_train, 
                X_test, 
                y_test
            )

            model_name: str = sorted(report.items(), key=lambda kv: kv[1][1])[::-1][0][0]
            best_score: float = sorted(report.items(), key=lambda kv: kv[1][1])[::-1][0][1][1]
            best_model = models[model_name]

            if best_score < 0.65:
                raise CustomException("No best model found")
            
            logging.info(
                "%s produced an adjusted R² of %s on the test set", 
                model_name, 
                np.round(best_score, 2)
            )

            logging.info(
                "Saving the %s model as './%s'", 
                model_name,
                self.train_config.model_path
            )
            save_artifact(
                self.train_config.model_path, 
                best_model
            )

            print(f"{model_name} produced an adjusted R² of {best_score:.2f} on the test set.")
        except Exception as err:
            raise CustomException(err, sys)
        

if __name__=="__main__":
    train_data_path, test_data_path = DataIngestion().initiate_ingestion()
    df_train, df_test, _ = DataTransformation().initiate_transformation(train_data_path, test_data_path)
    Train().initiate_train(df_train, df_test)
