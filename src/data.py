from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    raw: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    feature_columns: list[str]


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
# ``train.csv`` is retained for the original Kaggle workflow.  The bundled
# dataset uses ``price`` as its target and is detected automatically as well.
PROJECT_DATASET_PATHS = (
    DATA_DIRECTORY / "train.csv",
    DATA_DIRECTORY / "house_price_50k.csv",
)


def load_project_dataset() -> pd.DataFrame | None:
    """Load the first supported CSV found in the project's data directory."""
    for path in PROJECT_DATASET_PATHS:
        if path.is_file():
            return pd.read_csv(path)
    return None


def make_demo_data(n: int = 500, seed: int = 17) -> pd.DataFrame:
    """Create clearly labeled synthetic Ames-like data for a no-download demo."""
    rng = np.random.default_rng(seed)
    quality = rng.integers(2, 10, n)
    living_area = rng.normal(1550, 500, n).clip(500, 4500).round().astype(int)
    garage = rng.integers(0, 3, n)
    age = rng.integers(0, 100, n)
    neighborhood = rng.choice(["CollgCr", "OldTown", "NAmes", "NridgHt", "Edwards"], n)
    effect = pd.Series(neighborhood).map({"CollgCr": 9000, "OldTown": -12000, "NAmes": 0, "NridgHt": 27000, "Edwards": -7000}).to_numpy()
    price = (25000 + quality * 11500 + living_area * 82 + garage * 8500 - age * 430 + effect + rng.normal(0, 18000, n)).clip(45000)
    return pd.DataFrame({"OverallQual": quality, "GrLivArea": living_area, "GarageCars": garage, "HouseAge": age, "Neighborhood": neighborhood, "SalePrice": price.round().astype(int)})


def prepare_data(raw: pd.DataFrame) -> Dataset:
    # The included 50k-house dataset names its target ``price``.  Normalize it
    # internally so the rest of the app continues to use a single target name.
    if "SalePrice" not in raw.columns and "price" in raw.columns:
        raw = raw.rename(columns={"price": "SalePrice"})
    if "SalePrice" not in raw.columns:
        raise ValueError("The CSV must contain a numeric `SalePrice` or `price` target column.")
    if not pd.api.types.is_numeric_dtype(raw["SalePrice"]):
        raise ValueError("`SalePrice` must contain numeric house prices.")
    raw = raw.dropna(subset=["SalePrice"]).copy()
    raw = add_house_features(raw)
    y = raw["SalePrice"].astype(float)
    excluded = {"SalePrice", "Id"}
    features = [c for c in raw.columns if c not in excluded]
    if not features:
        raise ValueError("No usable feature columns were found.")
    X = raw[features].copy()
    for col in X.select_dtypes(include=["datetime", "datetimetz"]).columns:
        X[col] = X[col].astype("int64")
    return Dataset(raw=raw, X=X, y=y, feature_columns=features)


def add_house_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add transparent domain aggregates while retaining all source features."""
    data = frame.copy()

    def numeric(name: str) -> pd.Series:
        values = data.get(name)
        if values is None:
            return pd.Series(0.0, index=data.index)
        return pd.to_numeric(values, errors="coerce").fillna(0)

    if {"YrSold", "YearBuilt"}.issubset(data.columns):
        data["HouseAgeAtSale"] = (numeric("YrSold") - numeric("YearBuilt")).clip(lower=0)
    if {"YrSold", "YearRemodAdd"}.issubset(data.columns):
        data["YearsSinceRemodel"] = (numeric("YrSold") - numeric("YearRemodAdd")).clip(lower=0)
    if {"YearBuilt", "YearRemodAdd"}.issubset(data.columns):
        data["IsRemodeled"] = (numeric("YearRemodAdd") > numeric("YearBuilt")).astype(int)

    areas = [name for name in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"] if name in data.columns]
    if areas:
        data["TotalFinishedArea"] = sum((numeric(name) for name in areas), start=pd.Series(0.0, index=data.index))

    outdoors = [name for name in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"] if name in data.columns]
    if outdoors:
        data["TotalOutdoorArea"] = sum((numeric(name) for name in outdoors), start=pd.Series(0.0, index=data.index))

    bathrooms = {name for name in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"] if name in data.columns}
    if bathrooms:
        data["TotalBathrooms"] = numeric("FullBath") + numeric("BsmtFullBath") + 0.5 * numeric("HalfBath") + 0.5 * numeric("BsmtHalfBath")
    return data
