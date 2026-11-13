import numpy as np
import pandas as pd
import shap


def explain_prediction(model, record: pd.DataFrame):
    """Return grouped SHAP contributions and the model baseline for one house."""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    transformed = preprocessor.transform(record)
    names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(regressor)
    values = np.asarray(explainer.shap_values(transformed)).reshape(-1)

    grouped = {}
    for name, value in zip(names, values):
        raw_name = name.split("__", 1)[-1]
        raw_name = next((column for column in record.columns if raw_name == column or raw_name.startswith(f"{column}_")), raw_name)
        grouped[raw_name] = grouped.get(raw_name, 0) + value

    baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return pd.Series(grouped).sort_values(key=abs, ascending=False), baseline
