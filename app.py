from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.data import load_project_dataset, make_demo_data, prepare_data
from src.explain import explain_record, stability_experiment, top_contributions
from src.modeling import evaluate_model, fit_model


st.set_page_config(page_title="House Price Explainer", page_icon="🏠", layout="wide")

st.title("House Price Explainer")
st.caption("Estimate a house sale price, inspect its SHAP explanation, and test explanation stability.")


@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    return make_demo_data()


@st.cache_data(show_spinner="Preparing the data and training the model…")
def train(csv_bytes: bytes | None, seed: int):
    project_data = load_project_dataset()
    raw = pd.read_csv(io.BytesIO(csv_bytes)) if csv_bytes else (project_data if project_data is not None else load_demo())
    dataset = prepare_data(raw)
    model, x_train, x_test, y_train, y_test, preprocessor = fit_model(dataset, seed)
    metrics = evaluate_model(model, x_test, y_test)
    return dataset, model, x_test, y_test, preprocessor, metrics


with st.sidebar:
    st.header("Data and model")
    uploaded = st.file_uploader("Upload Kaggle train.csv (optional)", type=["csv"])
    st.caption("Using data/house_price_50k.csv. You can also upload a temporary CSV with a `price` or `SalePrice` target.")
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    st.divider()
    st.markdown("**Model:** Random Forest Regressor  \n**Validation:** 20% holdout split")

csv_bytes = uploaded.getvalue() if uploaded else None
try:
    dataset, model, x_test, y_test, preprocessor, metrics = train(csv_bytes, int(seed))
except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.info("Expected a CSV containing a numeric `price` or `SalePrice` column.")
    st.stop()

if uploaded is None and load_project_dataset() is None:
    st.warning("Demo mode: results use generated sample data and are not meaningful real estate valuations.")
elif uploaded is None:
    st.success("Using the project dataset found in the data folder.")

with st.container(border=True):
    st.subheader("Model performance on the holdout set")
    a, b, c = st.columns(3)
    a.metric("Mean Absolute Error", f"${metrics['mae']:,.0f}")
    b.metric("Root Mean Squared Error", f"${metrics['rmse']:,.0f}")
    c.metric("Holdout records", f"{len(x_test):,}")

left, right = st.columns([0.95, 1.2], gap="large")
with left:
    st.subheader("Choose a property")
    row_id = st.selectbox("Holdout property", options=list(x_test.index), format_func=lambda i: f"Property {i}")
    record = x_test.loc[[row_id]].copy()
    prediction = float(model.predict(record)[0])
    st.metric("Predicted sale price", f"${prediction:,.0f}")
    if "SalePrice" in dataset.raw.columns:
        actual = float(y_test.loc[row_id])
        st.caption(f"Actual sale price in the holdout data: ${actual:,.0f}")
    with st.expander("Property features"):
        st.dataframe(record.T.rename(columns={row_id: "Value"}), use_container_width=True)

with right:
    st.subheader("Why did the model make this prediction?")
    try:
        grouped, base_value = explain_record(model, preprocessor, record, dataset.feature_columns)
        st.caption(f"Baseline prediction: ${base_value:,.0f}. Bars show the feature contribution to this prediction.")
        positive, negative = top_contributions(grouped, record)
        up, down = st.columns(2)
        with up:
            st.markdown("#### Features raising the estimate")
            if positive.empty:
                st.caption("No positive contribution for this record.")
            for feature, row in positive.iterrows():
                st.metric(feature, str(row["Property value"]), delta=f"+${row['SHAP contribution ($)']:,.0f}")
        with down:
            st.markdown("#### Features lowering the estimate")
            if negative.empty:
                st.caption("No negative contribution for this record.")
            for feature, row in negative.iterrows():
                st.metric(feature, str(row["Property value"]), delta=f"${row['SHAP contribution ($)']:,.0f}", delta_color="inverse")
        st.markdown("#### Full contribution ranking")
        st.bar_chart(grouped.rename("SHAP contribution ($)"), horizontal=True, color="#3867D6")
        st.dataframe(grouped.rename("Contribution ($)").to_frame(), use_container_width=True)
        st.caption("Positive values raise the prediction relative to the model baseline; negative values lower it.")
    except ImportError:
        st.error("SHAP is not installed. Install dependencies from requirements.txt.")
    except Exception as exc:
        st.error(f"Could not calculate the SHAP explanation: {exc}")

st.divider()
st.subheader("Stability experiment")
st.write("Perturb one numeric feature by a small percentage and compare the prediction and SHAP contributions.")
numeric_features = [c for c in dataset.feature_columns if pd.api.types.is_numeric_dtype(dataset.X[c])]
if not numeric_features:
    st.info("No numeric features are available for the stability experiment.")
else:
    feature = st.selectbox("Feature to perturb", numeric_features, index=0)
    change_pct = st.slider("Change from the selected property's value", min_value=-10, max_value=10, value=5, step=1, format="%d%%")
    try:
        result = stability_experiment(model, preprocessor, record, feature, change_pct, dataset.feature_columns)
        x1, x2 = st.columns(2)
        x1.metric("Original prediction", f"${result['original_prediction']:,.0f}")
        x2.metric("Perturbed prediction", f"${result['perturbed_prediction']:,.0f}", delta=f"${result['prediction_delta']:,.0f}")
        st.dataframe(result["comparison"], use_container_width=True)
        st.caption("SHAP values describe this fitted model; this small perturbation experiment does not establish causal effects or guarantee price accuracy.")
    except Exception as exc:
        st.error(f"Could not run the stability experiment: {exc}")

with st.expander("Methodology and limitations"):
    st.markdown("""
    - Categorical fields are one-hot encoded; missing values are imputed inside the training pipeline.
    - The model is a Random Forest Regressor trained on a random training split.
    - MAE and RMSE summarize errors on the held-out records; they do not measure uncertainty for an individual home.
    - SHAP attributes the model output across input features. It does not prove that a feature causes a price change.
    - The stability experiment changes one numeric input at a time while holding all other values fixed.
    - The Kaggle dataset reflects a specific market and period; predictions should not be used as professional valuations.
    """)
