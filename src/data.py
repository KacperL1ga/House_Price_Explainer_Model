from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "house_price_50k.csv"
TARGET = "price"
FEATURES = [
    "area", "bedrooms", "bathrooms", "floors", "age", "distance",
    "garage", "parking", "garden", "security", "school_nearby",
    "hospital_nearby", "shopping_mall_nearby", "public_transport",
    "crime_rate", "population_density", "location", "income_level",
]


def load_data() -> pd.DataFrame:
    """Load the project dataset and keep only the model columns."""
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {', '.join(missing)}")
    return data[required].dropna()
