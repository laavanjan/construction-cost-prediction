import streamlit as st
import joblib
import pandas as pd

# Load preprocessor
@st.cache_resource
def load_preprocessor(path="preprocessor.pkl"):
    return joblib.load(path)

# Load trained best model
@st.cache_resource
def load_best_model(path="best_model.pkl"):
    return joblib.load(path)

preprocessor = load_preprocessor()
best_model = load_best_model()

st.title("🏗️ Construction Cost Prediction App")

# User input form
st.sidebar.header("Enter Project Details")

building_type = st.sidebar.selectbox(
    "Building Type", ["Residential", "Commercial", "Industrial"]
)
area_sqm = st.sidebar.slider("Area (sq.m)", 250, 15000, value=2000, step=50)
floors = st.sidebar.slider("Number of Floors", 1, 20, value=2)
location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
foundation_type = st.sidebar.selectbox("Foundation Type", ["Concrete", "Pile", "Slab"])
roof_type = st.sidebar.selectbox("Roof Type", ["Flat", "Pitched", "Dome"])

# Use Yes/No instead of 0/1 and convert to binary
has_basement = st.sidebar.selectbox("Has Basement?", ["No", "Yes"])
has_basement = 1 if has_basement == "Yes" else 0

has_parking = st.sidebar.selectbox("Has Parking?", ["No", "Yes"])
has_parking = 1 if has_parking == "Yes" else 0

labor_rate = st.sidebar.slider("Labor Rate (per hour)", 3000, 15000, value=8000, step=100)

# Create a DataFrame for user input
input_data = pd.DataFrame({
    "building_type": [building_type],
    "area_sqm": [area_sqm],
    "floors": [floors],
    "location": [location],
    "foundation_type": [foundation_type],
    "roof_type": [roof_type],
    "has_basement": [has_basement],
    "has_parking": [has_parking],
    "labor_rate": [labor_rate]
})

# Preprocess user input
def preprocess_input(data, preprocessor):
    return preprocessor.transform(data)

if st.button("Predict Cost"):
    try:
        # Display input data for debugging
        st.subheader("Input Data")
        st.dataframe(input_data)
        
        # Transform input data
        transformed_input = preprocess_input(input_data, preprocessor)
        st.write(f"Transformed input shape: {transformed_input.shape}")

        # Make prediction using best_model
        prediction = best_model.predict(transformed_input)
        predicted_cost = prediction[0]
        
        # Display result
        if predicted_cost > 0:
            st.success(f"🏗️ Estimated Construction Cost: ${predicted_cost:,.2f}")
            
            # Add cost breakdown information
            cost_per_sqm = predicted_cost / area_sqm
            st.info(f"📊 Cost per sq.m: ${cost_per_sqm:,.2f}")
        else:
            st.error(f"⚠️ Model predicted negative cost: ${predicted_cost:,.2f}")
            st.warning("This may indicate an issue with the input values or model training.")
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.exception(e)  # Show full traceback for debugging
