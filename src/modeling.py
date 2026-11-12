from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import Dataset


def fit_model(dataset: Dataset, seed: int = 42):
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=0.2, random_state=seed
    )
    numeric = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in x_train.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical))
    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=True)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=seed)),
    ])
    model.fit(x_train, y_train)
    return model, x_train, x_test, y_train, y_test, preprocessor


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
    }
