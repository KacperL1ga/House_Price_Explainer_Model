from time import perf_counter
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data import FEATURES, TARGET, load_data
from src.modeling import make_preprocessor


def models(seed: int):
    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "Extra Trees": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "XGBRegressor": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.85,
            colsample_bytree=0.9, objective="reg:squarederror", tree_method="hist",
            n_jobs=-1, random_state=seed, verbosity=0)
    }
    return candidates


def run_benchmark(seed: int = 42) -> pd.DataFrame:
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=seed
    )
    results = []
    for name, regressor in models(seed).items():
        model = Pipeline([("preprocessor", make_preprocessor()), ("regressor", regressor)])
        started = perf_counter()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results.append({
            "Model": name,
            "R2": r2_score(y_test, prediction),
            "MAE": mean_absolute_error(y_test, prediction),
            "RMSE": mean_squared_error(y_test, prediction) ** 0.5,
            "Czas treningu (s)": perf_counter() - started,
        })
    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)


if __name__ == "__main__":
    result = run_benchmark()
    result.to_csv("benchmark_results.csv", index=False)
    print(result.to_string(index=False))
    print("\nRanking zapisano w benchmark_results.csv")
