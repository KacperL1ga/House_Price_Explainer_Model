import pandas as pd
import streamlit as st

from src.data import FEATURES, load_data
from src.explain import explain_prediction
from src.modeling import train_model


LABELS = {
    "area": "Floor area", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms",
    "floors": "Floors", "age": "Property age", "distance": "Distance to city centre",
    "garage": "Garage", "parking": "Parking space", "garden": "Garden",
    "security": "Security service", "school_nearby": "School nearby",
    "hospital_nearby": "Hospital nearby", "shopping_mall_nearby": "Shopping mall nearby",
    "public_transport": "Public transport nearby", "crime_rate": "Crime rate",
    "population_density": "Population density", "location": "Location quality",
    "income_level": "Area income level",
}


st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="wide")
st.title("🏠 House Price Estimator")
st.write("Enter the property details to receive an estimated sale price and a SHAP explanation.")


@st.cache_resource(show_spinner="Training XGBoost on 50,000 property records...")
def load_model():
    return train_model(load_data(), seed=42)


model, _, _, metrics = load_model()
with st.expander("Model performance"):
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Mean Absolute Error", f"${metrics['mae']:,.0f}")
    metric_2.metric("Root Mean Squared Error", f"${metrics['rmse']:,.0f}")
    metric_3.metric("R² score", f"{metrics['r2']:.3f}")

st.divider()
st.subheader("Property details")
st.caption("Change any value by a small step and submit the form again to check how the prediction responds.")

with st.form("house_form"):
    col_1, col_2, col_3 = st.columns(3)

    with col_1:
        st.markdown("#### Building")
        area = st.number_input("Floor area (m²)", min_value=300, max_value=5000, value=1500, step=50, help="Total usable floor area of the property in square metres.")
        bedrooms = st.number_input("Bedrooms (count)", min_value=1, max_value=10, value=3, step=1, help="Number of bedrooms in the property.")
        bathrooms = st.number_input("Bathrooms (count)", min_value=1, max_value=6, value=2, step=1, help="Number of bathrooms in the property.")
        floors = st.number_input("Floors (count)", min_value=1, max_value=4, value=2, step=1, help="Number of floors in the property.")
        age = st.number_input("Property age (years)", min_value=0, max_value=120, value=20, step=1, help="Age of the property in years.")
        distance = st.number_input("Distance to city centre (km)", min_value=0, max_value=50, value=10, step=1, help="Distance from the property to the city centre in kilometres.")

    with col_2:
        st.markdown("#### Amenities")
        garage = st.selectbox("Garage", options=[0, 1], format_func=lambda value: "Yes" if value else "No", help="Whether the property includes a garage.")
        parking = st.selectbox("Parking space", options=[0, 1], format_func=lambda value: "Yes" if value else "No", help="Whether the property includes a dedicated parking space.")
        garden = st.selectbox("Garden", options=[0, 1], format_func=lambda value: "Yes" if value else "No", help="Whether the property includes a private garden.")
        security = st.selectbox("Security service", options=[0, 1], format_func=lambda value: "Yes" if value else "No", help="Whether the property or its estate has a security service.")
        school_nearby = st.selectbox("School nearby", options=[0, 1], format_func=lambda value: "Yes" if value else "No", help="Whether a school is located near the property.")
        hospital_nearby = st.selectbox("Hospital nearby", options=[0, 1], format_func=lambda value: "Yes" if value else "No", help="Whether a hospital is located near the property.")
        shopping_mall_nearby = st.selectbox("Shopping mall nearby", options=[0, 1], format_func=lambda value: "Yes" if value else "No", help="Whether a shopping mall is located near the property.")
        public_transport = st.selectbox("Public transport nearby", options=[0, 1], format_func=lambda value: "Yes" if value else "No", help="Whether public transport is easily accessible from the property.")

    with col_3:
        st.markdown("#### Neighbourhood")
        crime_rate = st.number_input("Crime rate (0–10)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, help="Area crime-rate score used in the dataset; 0 is lowest and 10 is highest.")
        population_density = st.number_input("Population density (people/km²)", min_value=500, max_value=15000, value=5000, step=100, help="Population density of the surrounding area in people per square kilometre.")
        location = st.selectbox("Location quality", options=["low", "medium", "premium"], format_func=lambda value: {"low": "Low", "medium": "Medium", "premium": "Premium"}[value], help="Overall location tier assigned in the dataset.")
        income_level = st.selectbox("Area income level", options=["low", "mid", "high"], format_func=lambda value: {"low": "Low", "mid": "Middle", "high": "High"}[value], help="Income-level category of the surrounding area in the dataset.")

    submitted = st.form_submit_button("Estimate property price", type="primary", width = 'stretch')

if submitted:
    values = [
        area, bedrooms, bathrooms, floors, age, distance, garage, parking, garden,
        security, school_nearby, hospital_nearby, shopping_mall_nearby,
        public_transport, crime_rate, population_density, location, income_level,
    ]
    house = pd.DataFrame([values], columns=FEATURES)
    prediction = float(model.predict(house)[0])
    contributions, baseline = explain_prediction(model, house)

    st.divider()
    st.subheader("Estimated value")
    st.metric("Estimated sale price", f"${prediction:,.0f}")
    st.caption(f"SHAP baseline: ${baseline:,.0f}. Contributions show how each feature moves this prediction from the baseline.")

    positive = contributions[contributions > 0].sort_values(ascending=False).head(5)
    negative = contributions[contributions < 0].sort_values().head(5)
    up, down = st.columns(2)
    with up:
        st.markdown("#### Top 5 factors increasing the estimate")
        if positive.empty:
            st.info("No positive SHAP contributions for this property.")
        for feature, impact in positive.items():
            st.metric(LABELS.get(feature, feature), str(house[feature].iloc[0]), delta=f"+${impact:,.0f}")
    with down:
        st.markdown("#### Top 5 factors decreasing the estimate")
        if negative.empty:
            st.info("No negative SHAP contributions for this property.")
        for feature, impact in negative.items():
            st.metric(LABELS.get(feature, feature), str(house[feature].iloc[0]), delta=f"${impact:,.0f}", delta_color="inverse")

    st.markdown("#### Full SHAP contribution ranking")
    st.bar_chart(contributions.rename("Impact on price ($)"), horizontal=True, color="#3867D6")

with st.expander("How to interpret the metrics"):
    st.markdown("""
    - **MAE** is the average price error in dollars.
    - **RMSE** penalises occasional large errors more strongly.
    - **R²** shows the share of price variation explained by the model; values closer to 1 are better.
    - SHAP explains model behaviour. It does not prove that a feature causes a price change.
    """)
