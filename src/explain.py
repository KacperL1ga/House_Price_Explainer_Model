from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _encoded_to_raw_map(encoded_names: list[str], raw_features: list[str]) -> list[str]:
    ordered = sorted(raw_features, key=len, reverse=True)
    mapping = []
    for name in encoded_names:
        plain = name.split("__", 1)[-1]
        source = next((f for f in ordered if plain == f or plain.startswith(f + "_")), plain)
        mapping.append(source)
    return mapping


def _values(model, preprocessor, record: pd.DataFrame, raw_features: list[str]):
    transformed = preprocessor.transform(record)
    estimator = model.named_steps["regressor"]
    names = preprocessor.get_feature_names_out().tolist()
    explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3:
        values = values[0]
    values = values.reshape(-1)
    grouped = pd.Series(values, index=_encoded_to_raw_map(names, raw_features)).groupby(level=0).sum()
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return grouped.sort_values(key=lambda s: s.abs(), ascending=False), base


def explain_record(model, preprocessor, record: pd.DataFrame, raw_features: list[str]):
    return _values(model, preprocessor, record, raw_features)


def top_contributions(grouped: pd.Series, record: pd.DataFrame, count: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the strongest positive and negative SHAP contributions for the UI."""
    table = pd.DataFrame({"SHAP contribution ($)": grouped})
    table["Property value"] = [record[column].iloc[0] if column in record.columns else "derived" for column in table.index]
    positive = table[table["SHAP contribution ($)"] > 0].head(count)
    negative = table[table["SHAP contribution ($)"] < 0].sort_values("SHAP contribution ($)").head(count)
    return positive, negative


def stability_experiment(model, preprocessor, record: pd.DataFrame, feature: str, change_pct: int, raw_features: list[str]):
    original = record.copy()
    perturbed = record.copy()
    perturbed.loc[:, feature] = pd.to_numeric(perturbed[feature], errors="coerce") * (1 + change_pct / 100)
    p0 = float(model.predict(original)[0])
    p1 = float(model.predict(perturbed)[0])
    s0, _ = _values(model, preprocessor, original, raw_features)
    s1, _ = _values(model, preprocessor, perturbed, raw_features)
    all_features = sorted(set(s0.index) | set(s1.index))
    comparison = pd.DataFrame({
        "Original SHAP ($)": s0.reindex(all_features, fill_value=0.0),
        "Perturbed SHAP ($)": s1.reindex(all_features, fill_value=0.0),
    })
    comparison["Change ($)"] = comparison["Perturbed SHAP ($)"] - comparison["Original SHAP ($)"]
    comparison = comparison.reindex(comparison["Change ($)"].abs().sort_values(ascending=False).index)
    return {"original_prediction": p0, "perturbed_prediction": p1, "prediction_delta": p1 - p0, "comparison": comparison}
