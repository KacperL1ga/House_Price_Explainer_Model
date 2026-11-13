import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.data import FEATURES, TARGET


def make_preprocessor() -> ColumnTransformer:
    numeric = FEATURES[:-2]
    categorical = FEATURES[-2:]
    return ColumnTransformer([
        ("numbers", SimpleImputer(strategy="median"), numeric),
        ("categories", Pipeline([
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def train_model(data: pd.DataFrame, seed: int = 42):
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("regressor", XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )),
    ])
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, prediction),
        "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
        "r2": r2_score(y_test, prediction),
    }
    return model, X_test, y_test, metrics
