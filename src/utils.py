"""
Module containing helper functions
"""
from typing import Tuple, Dict, Any
import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd

from yaml.loader import SafeLoader
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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
        X_train, y_train = train_set.drop(target, axis=1), train_set[target]
        X_test, y_test = test_set.drop(target, axis=1), test_set[target]
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

        # concatenate the train and test set imputed feature matrix and target vector
        train_set_imputed = (pd.concat
                             ([X_train_imputed, y_train], axis=1)
                             .reset_index(drop=True)
                             )
        test_set_imputed = (pd.concat
                            ([X_test_imputed, y_test], axis=1)
                            .reset_index(drop=True)
                            )
        return train_set_imputed, test_set_imputed
    except Exception as err:
        raise CustomException(err, sys)


def adj_rsquared(
        feature_matrix: pd.DataFrame,
        target_vector: pd.Series,
        prediction_vector: np.ndarray
) -> float:
    """
    Calculates and returns the adjusted R²
    Args:
        feature_matrix: Feature matrix of shape (N, D)
        target_vector: Target vector of length N
        prediction_vector: Prediction vector of length N
    Returns:
        adj_r2: Adjusted R²
    """
    try:
        n_records, n_features = feature_matrix.shape
        total = target_vector - np.mean(target_vector)
        ss_total = total.dot(total)
        error = target_vector - prediction_vector
        ss_error = error.dot(error)
        r2_score = 1 - (ss_error / ss_total)
        adj_r2 = 1 - (((1 - r2_score) * (n_records - 1)) / (n_records - n_features - 1))
        return adj_r2
    except Exception as err:
        raise CustomException(err, sys)


def evaluate_models(
        models: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
) -> Tuple[str, Any, float]:
    """
    Trains and evaluates several regressors

    Args:
        models: Dictionary of regressors
        X_train: ML-ready train set feature matrix
        y_train: ML-ready train set target vector
        X_test: ML-ready test set feature matrix
        y_test: ML-ready test set target vector

    Returns:
        model_name: Name of the regressor that
        produced the highest test set adjusted R²
        best_model: The regressor that corresponds
        to model_name
        best_score: Test set adjusted R² that
        best_model produced
    """
    try:
        params = get_params(r"./conf/parameters.yml")
        report = {}
        for name, model in models.items():
            if name == "LinearRegression":
                model.fit(X_train, y_train)
                train_predictions = model.predict(X_train)
                train_metric = adj_rsquared(X_train, y_train, train_predictions)
                test_predictions = model.predict(X_test)
                test_metric = adj_rsquared(X_test, y_test, test_predictions)
                report[name] = [model, train_metric, test_metric]
            else:
                gscv = GridSearchCV(
                    estimator=model,
                    param_grid=params["grid_search_cv"]["param_grid"][name],
                    scoring=params["grid_search_cv"]["scoring"],
                    refit=params["grid_search_cv"]["refit"],
                    cv=params["grid_search_cv"]["cv"],
                    n_jobs=params["grid_search_cv"]["n_jobs"]
                )
                gscv.fit(X_train, y_train)
                train_predictions = gscv.predict(X_train)
                train_metric = adj_rsquared(X_train, y_train, train_predictions)
                test_predictions = gscv.predict(X_test)
                test_metric = adj_rsquared(X_test, y_test, test_predictions)
                report[name] = [gscv.best_estimator_, train_metric, test_metric]

        model_name: str = sorted(report.items(), key=lambda kv: kv[1][-1])[::-1][0][0]
        best_model = sorted(report.items(), key=lambda kv: kv[1][-1])[::-1][0][1][0]
        best_score: float = sorted(report.items(), key=lambda kv: kv[1][-1])[::-1][0][1][-1]
        return model_name, best_model, best_score
    except Exception as err:
        raise CustomException(err, sys)


def save_artifact(artifact_path: str, artifact):
    """
    Writes artifact to artifact_path

    Args:
        artifact_path: File path the artifact is written to
        artifact: Python object
    """
    try:
        directory = os.path.dirname(artifact_path)
        os.makedirs(directory, exist_ok=True)
        if artifact_path[-3:] == "csv":
            artifact.to_csv(artifact_path, index=False)
        elif artifact_path[-3:] == "pkl":
            with open(artifact_path, "wb") as fp:
                pickle.dump(artifact, fp)
    except Exception as err:
        raise CustomException(err, sys)
