# House Price Explainer — project report draft

## 1. Objective

This project builds an interactive application that predicts house sale prices and explains individual model predictions. It also investigates how a small change to one input feature changes the prediction and its SHAP explanation.

## 2. Dataset

The planned dataset is the training file from Kaggle's *House Prices: Advanced Regression Techniques* competition. It contains property features and `SalePrice`, the target to predict. The application also includes synthetic data only to demonstrate the interface; do not use demo-mode metrics as project findings.

**Complete after loading the Kaggle file:**

- Number of records used: [fill in]
- Number of input columns after excluding `Id` and `SalePrice`: [fill in]
- Missing-value summary and any additional data decisions: [fill in]

## 3. Methodology

Rows with missing target values are excluded. The data is split into 80% training and 20% holdout sets using a fixed random seed. Numeric missing values are imputed with the training median. Categorical missing values are imputed with the training mode and categories are one-hot encoded. A Random Forest Regressor is fitted to the training set. The model is evaluated on the holdout set using mean absolute error (MAE) and root mean squared error (RMSE).

The model keeps all supplied attributes and adds transparent calculated features: age at sale, time since renovation, a renovation flag, aggregate indoor area, aggregate outdoor area, and weighted bathroom count. This preserves the original dataset while making meaningful quantities easier for the model to learn.

TreeSHAP values are used to explain a selected holdout prediction relative to the fitted model's expected output. For the stability experiment, one numeric feature is changed by a chosen percentage from -10% to +10%, with all other inputs held fixed. Both the predicted price and grouped SHAP contributions are compared before and after the change.

## 4. Results

The final numeric results must be recorded after uploading the Kaggle training file. They are intentionally left blank here rather than inventing measurements.

| Measure | Holdout result |
| --- | ---: |
| MAE | [fill in from app] |
| RMSE | [fill in from app] |
| Holdout size | [fill in from app] |

**Example property explanation:** [Describe the selected record, predicted price, baseline, and the strongest positive and negative feature contributions shown by the app.]

**Stability experiment:** [Record the chosen property, feature, percentage change, prediction change, and the largest changes in SHAP contributions.]

## 5. Discussion and limitations

MAE expresses the average absolute prediction error in the target's currency units. RMSE gives greater weight to large errors. Neither metric describes the uncertainty of an individual prediction. A random holdout split can also make evaluation optimistic if records share temporal or geographic patterns.

SHAP contributions explain the fitted model's output; they do not establish that changing a property feature causes its market price to change. The one-feature experiment is a local sensitivity check, not a causal analysis. Results are limited to the dataset's specific market, time period, and feature coverage. The model and application are educational and must not be presented as a professional appraisal or a guarantee of a sale price.

## 6. Conclusion

[Complete after reviewing the holdout metrics, property explanations, and stability experiment. State what the model appears to capture, where explanations change under perturbation, and what further validation would be needed before relying on the results.]
