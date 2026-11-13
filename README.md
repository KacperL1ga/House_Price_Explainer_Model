# House Price Estimator

A Streamlit application that estimates house prices from `house_price_50k.csv`.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Project files

- `app.py` — property form, price estimate, and SHAP explanation.
- `src/data.py` — dataset path and the 18 model features.
- `src/modeling.py` — XGBoost training and MAE, RMSE, and R² metrics.
- `src/explain.py` — local SHAP explanation for the submitted property.
- `src/benchmark.py` — model comparison script.

## Benchmark

```powershell
python -m src.benchmark
```

The script compares Random Forest, Extra Trees, HistGradientBoosting, and XGBoost on the same 80/20 split. Results are saved to `benchmark_results.csv`. The application uses XGBoost because it achieved the best benchmark result.

RMSE is the main selection metric because it is expressed in dollars and penalises large errors. MAE and R² provide additional context.
