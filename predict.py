import streamlit as st
import joblib

import pandas as pd

# Load trained model
@st.cache_resource
def load_model(path='cost_model.pkl'):
    return joblib.load(path)

model = load_model()

st.title("🏗️ Construction Cost Prediction App")

# User input form
st.sidebar.header("Enter Project Details")

building_type = st.sidebar.selectbox("Building Type", ['Residential', 'Commercial', 'Industrial'])
area_sqm = st.sidebar.slider("Area (sq.m)", 100, 10000, step=50)
floors = st.sidebar.slider("Number of Floors", 1, 50)
location = st.sidebar.selectbox("Location", ['Urban', 'Suburban', 'Rural'])
quality_grade = st.sidebar.selectbox("Quality Grade", ['Standard', 'Premium', 'Luxury'])
foundation_type = st.sidebar.selectbox("Foundation Type", ['Concrete', 'Pile', 'Slab'])
roof_type = st.sidebar.selectbox("Roof Type", ['Flat', 'Pitched', 'Dome'])

# Use Yes/No instead of 0/1 and convert to binary
has_basement = st.sidebar.selectbox("Has Basement?", ['No', 'Yes'])
has_basement = 1 if has_basement == 'Yes' else 0

has_elevator = st.sidebar.selectbox("Has Elevator?", ['No', 'Yes'])
has_elevator = 1 if has_elevator == 'Yes' else 0

has_parking = st.sidebar.selectbox("Has Parking?", ['No', 'Yes'])
has_parking = 1 if has_parking == 'Yes' else 0

# Labor rate in Sri Lankan Rupees (LKR)
labor_rate = st.sidebar.slider("Labor Rate (LKR/hour)", 100.0, 1500.0, step=50.0)

# Material cost index
material_cost_index = st.sidebar.slider("Material Cost Index", 0.8, 1.5, step=0.01)


# Prepare input DataFrame
input_data = pd.DataFrame([{
    'building_type': building_type,
    'area_sqm': area_sqm,
    'floors': floors,
    'location': location,
    'quality_grade': quality_grade,
    'foundation_type': foundation_type,
    'roof_type': roof_type,
    'has_basement': has_basement,
    'has_elevator': has_elevator,
    'has_parking': has_parking,
    'labor_rate': labor_rate,
    'material_cost_index': material_cost_index
}])

# Predict
if st.button("Predict Cost"):
    predicted_cost = model.predict(input_data)[0]
    st.success(f"🏷️ Estimated Construction Cost: Rs {predicted_cost:,.2f}")
