# House Price Explainer

Interactive Streamlit project for house price prediction, local SHAP explanations, and a small explanation-stability experiment.

## Run locally

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The application automatically uses the included `data/house_price_50k.csv` dataset (50,000 property records). Its `price` column is recognized as the target automatically. Uploading a CSV in the sidebar temporarily takes precedence; uploaded data may use either `price` or `SalePrice` as a numeric target. For the original Kaggle workflow, a `data/train.csv` file is also detected automatically and takes precedence when present.

## Project contents

- `app.py` — Streamlit interface, model metrics, property selection, SHAP view, and stability experiment.
- `src/data.py` — data validation, dataset auto-detection, a clearly labeled synthetic demo-data generator, and domain-based feature engineering.
- `src/modeling.py` — missing-value preprocessing, one-hot encoding, holdout split, Random Forest, MAE, and RMSE.
- `src/explain.py` — per-record TreeSHAP values and a one-feature perturbation comparison.
- `REPORT.md` — report draft with methodology and results fields to complete after uploading the Kaggle data.

## Method

The app reserves 20% of the rows for a holdout set using a fixed random seed. Numeric missing values are imputed with the training median; categorical values use the most frequent training value and one-hot encoding. A Random Forest Regressor is fitted on the remaining rows. MAE and RMSE are reported on the holdout set. SHAP values explain one selected model prediction relative to the fitted model's baseline output. The stability experiment scales one numeric input by a user-selected percentage (from -10% to +10%), then compares both predictions and grouped SHAP contributions.

All original dataset features remain available to the model. The calculated features are: house age at sale (`YrSold - YearBuilt`), years since remodel, a remodel flag, total finished area (basement plus first and second floors), total outdoor area (deck and porches), and weighted bathrooms (full bathroom = 1, half bathroom = 0.5). These are transparent mathematical aggregates, not manually imposed price rules.

## Interpretation and limitations

This is an educational prototype, not a real-estate valuation service. SHAP values describe how this model allocates its prediction across the supplied features; they are not causal effects and do not guarantee that a price is correct. The Kaggle data represents a particular housing market and period. Report the dataset, split seed, metrics, and these limitations in any project write-up. A random holdout evaluation may also overstate performance when records share market or time patterns.

## Suggested report outline

1. Problem and objective
2. Dataset and target variable
3. Cleaning and preprocessing decisions
4. Random Forest model and train/holdout split
5. MAE and RMSE results
6. SHAP explanation of selected properties
7. Stability experiment setup and observations
8. Limitations, ethical use, and conclusions
