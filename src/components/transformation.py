from typing import Dict, Any
import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from mrmr import mrmr_regression 

from src.logger import logging
from src.exception import CustomException
from src.utils import get_params, impute_data, save_artifact
from src.components.ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    feature_transformer_path = os.path.join("artifacts", "feature_transformer.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        self.params: Dict[str, Any] = get_params(r"./conf/parameters.yml")

    def feature_transformer(self):
        try:
            ct = ColumnTransformer([
                (
                "numeric_features", 
                StandardScaler(), 
                self.params["numeric_features"]
                ), 
                (
                "nominal_features", 
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"), 
                self.params["nominal_features"]
                ), 
                (
                "ordinal_features", 
                OrdinalEncoder(categories=[self.params["ordinal"]["categories"]]), 
                self.params["ordinal"]["features"]
                )
                ])
            return ct
        except Exception as err:
            raise CustomException(err, sys)
        
    def initiate_transformation(self, train_data_path: str, test_data_path: str):
        try:
            logging.info("Reading in the train and test data")
            df_train = pd.read_csv(train_data_path)
            df_test = pd.read_csv(test_data_path)

            logging.info("Feature transformation initiated")
            df_train, df_test = impute_data(df_train, df_test)
            target = self.params["target"]
            X_train, y_train = df_train.drop(target, axis=1), df_train[target]
            X_test, y_test = df_test.drop(target, axis=1), df_test[target]

            ft = self.feature_transformer()
            ft.fit(X_train)
            
            ohe_categories = []
            for array in ft.transformers_[1][1].categories_:
                ohe_categories += [cat.lower() for cat in array.tolist()]

            features = [
                feature.replace(" ", "_").replace("/", "_") 
                for feature 
                in self.params["numeric_features"] + ohe_categories + self.params["ordinal"]["features"]
            ]

            X_train = pd.DataFrame(
                ft.transform(X_train), columns=features, index=X_train.index.tolist()
            )
            X_test = pd.DataFrame(
                ft.transform(X_test), columns=features, index=X_test.index.tolist()
            )
            
            rel_features = mrmr_regression(
                 X=X_train, 
                 y=y_train, 
                 K=int(len(features)/2), 
                 relevance="f", 
                 redundancy="c"
            )
            logging.info(
                "Most relevant/least redundant features: %s, and %s", 
                ', '.join(rel_features[:-1]),
                rel_features[-1]
            )
            
            train_set = pd.concat([X_train[rel_features], y_train], axis=1)
            test_set = pd.concat([X_test[rel_features], y_test], axis=1)
            logging.info("Feature transformation completed")

            logging.info(
                "Saving the feature transformer as './%s'", 
                self.transformation_config.feature_transformer_path
                )
            save_artifact(
                self.transformation_config.feature_transformer_path, 
                ft
            )

            return (
                train_set, 
                test_set, 
                self.transformation_config.feature_transformer_path
            )
        except Exception as err:
            raise CustomException(err, sys)


if __name__=="__main__":
    train_data_path, test_data_path = DataIngestion().initiate_ingestion()
    _, _, _ = DataTransformation().initiate_transformation(train_data_path, test_data_path)
    