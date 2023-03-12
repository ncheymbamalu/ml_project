from typing import Tuple, Dict, Any
import os
import sys
import yaml
import joblib
import pandas as pd

from yaml.loader import SafeLoader
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.exception import CustomException


def get_params(params_path: str) -> Dict[str, Any]:
  """
  Returns the parameters defined in params_path

  Args:
    params_path: ./conf/parameters.yml

  Returns:
    params: Parameters defined in params_path
  """
  try:
    with open(params_path) as filepath:
       params = yaml.load(filepath, Loader=SafeLoader)
       return params
  except Exception as err:
    raise CustomException(err, sys)


def impute_data(
        train_set: pd.DataFrame, test_set: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns the imputed train and test sets

    Args:
        train_set: Train set with missing values
        test_set: Test set with missing values

    Returns:
        train_set_imputed: Train set with imputed values
        test_set_imputed: Test set with imputed values
    """
    try:
        params = get_params(r"./conf/parameters.yml")
        target = params["target"]
        X_train = train_set.drop(target, axis=1)
        y_train = train_set[target]
        X_test = test_set.drop(target, axis=1)
        y_test = test_set[target]
        X_train_imputed, X_test_imputed = X_train.copy(deep=True), X_test.copy(deep=True)
        num_cols = params["numeric_features"]
        cat_cols = params["nominal_features"] + params["ordinal"]["features"]
        null_cols = [
            col
            for col in num_cols + cat_cols
            if X_train[col].isna().sum() > 0
        ]

        # label encode the non-null categorical features
        for cat_col in cat_cols:
            if cat_col not in null_cols:
                categories = sorted(set(X_train[cat_col]))
                cat_to_idx = dict(zip(categories, range(len(categories))))
                X_train[cat_col] = X_train[cat_col].map(cat_to_idx).astype("int")
                X_test[cat_col] = X_test[cat_col].map(cat_to_idx).astype("int")

        for col in null_cols:
            features = [
                feature
                for feature in num_cols + cat_cols
                if feature not in null_cols
            ]
            # categorical feature imputation
            if (col in cat_cols) or (len(set(X_train[col].dropna())) <= 10):
                features += [col]
                train = X_train[features].copy(deep=True)
                train_notna = train[train[col].notna()].copy(deep=True)
                train_isna = train[train[col].isna()].copy(deep=True)
                train_indices_to_impute = train_isna.index.tolist()
                model = RandomForestClassifier()
                model.fit(train_notna.drop(col, axis=1), train_notna[col])
                test = X_test[features].copy(deep=True)
                test_isna = test[test[col].isna()].copy(deep=True)
                test_indices_to_impute = test_isna.index.tolist()
                X_train_imputed.loc[train_indices_to_impute, col] = model.predict(train_isna.drop(col, axis=1))
                X_test_imputed.loc[test_indices_to_impute, col] = model.predict(test_isna.drop(col, axis=1))
            # numeric feature imputation
            else:
                features += [col]
                train = X_train[features].copy(deep=True)
                train_notna = train[train[col].notna()].copy(deep=True)
                train_isna = train[train[col].isna()].copy(deep=True)
                train_indices_to_impute = train_isna.index.tolist()
                model = RandomForestRegressor()
                model.fit(train_notna.drop(col, axis=1), train_notna[col])
                test = X_test[features].copy(deep=True)
                test_isna = test[test[col].isna()].copy(deep=True)
                test_indices_to_impute = test_isna.index.tolist()
                X_train_imputed.loc[train_indices_to_impute, col] = model.predict(train_isna.drop(col, axis=1))
                X_test_imputed.loc[test_indices_to_impute, col] = model.predict(test_isna.drop(col, axis=1))

        # concatenate the train and test set imputed features and target
        train_set_imputed = pd.concat([X_train_imputed, y_train], axis=1).reset_index(drop=True)
        test_set_imputed = pd.concat([X_test_imputed, y_test], axis=1).reset_index(drop=True)
        return train_set_imputed, test_set_imputed
    except Exception as err:
       raise CustomException(err, sys)
    

def save_artifact(artifact_path: str, artifact):
  """
  Writes artifact to artifact_path

  Args:
    artifact_path: File path the artifact is saved to
    artifact: Python object
  """
  try:
     directory = os.path.dirname(artifact_path)
     os.makedirs(directory, exist_ok=True)
     if artifact_path[-3:] == "csv":
        artifact.to_csv(artifact_path, index=False)
     elif artifact_path[-3:] == "pkl":
        joblib.dump(artifact, artifact_path)
  except Exception as err:
     raise CustomException(err, sys)
