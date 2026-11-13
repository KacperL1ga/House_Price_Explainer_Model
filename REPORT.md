# House Price Estimator — Report

## Objective

The model predicts the house sale price (`price`) from 18 building and neighbourhood features.

## Dataset

The project uses `data/house_price_50k.csv` with 50,000 property records. Features include floor area, room counts, age, amenities, location, and neighbourhood characteristics.

## Method

The data is split randomly into 80% training data and 20% test data. Numeric missing values are filled with the median. Categorical values are filled with the most common category and one-hot encoded. The benchmark selected XGBoost as the best model. It was compared with Random Forest, Extra Trees, and HistGradientBoosting.

## Results

| Metric | Test result |
| --- | ---: |
| MAE | [copy from the application] |
| RMSE | [copy from the application] |
| R² | [copy from the application] |

## Interpretation

SHAP shows which submitted values increase or decrease a prediction relative to the model baseline. It describes model behaviour and does not prove a causal relationship or replace a professional property valuation.
