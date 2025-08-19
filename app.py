import streamlit as st
import pandas as pd
import joblib

# Cache the model so it's loaded only once
@st.cache_resource
def load_predictor(model_path="renewable_energy_model.pkl"):
    deployment_package = joblib.load(model_path)
    return deployment_package

# Predictor class
class RenewableEnergyPredictor:
    def __init__(self, deployment_package):
        self.model = deployment_package['model']
        self.scaler = deployment_package['scaler']
        self.label_encoder = deployment_package['label_encoder']
        self.feature_names = deployment_package['feature_names']
        
    def predict(self, input_data: dict):
        input_df = pd.DataFrame([input_data])
        # Ensure column order
        input_df = input_df[self.feature_names]

        # Encode Region_ID
        if 'Region_ID' in input_df.columns and input_df['Region_ID'].dtype == 'object':
            input_df['Region_ID'] = self.label_encoder.transform(input_df['Region_ID'])
        
        # Scale
        input_scaled = self.scaler.transform(input_df)
        return self.model.predict(input_scaled)[0]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="India Renewable Energy Predictor", page_icon="⚡", layout="centered")
st.title("⚡ India Renewable Energy Capacity Predictor")
st.write("Predict renewable energy capacity (MW) based on key infrastructure, economic, and environmental factors.")

# Load predictor once
deployment_package = load_predictor()
predictor = RenewableEnergyPredictor(deployment_package)

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Input Features")

    region_id = st.text_input("Region ID (e.g., IN-MH)", "IN-MH")
    total_energy_demand = st.number_input("Total Energy Demand (MWh)", min_value=0, value=4000000)
    coal_generation = st.number_input("Coal Power Generation (MWh)", min_value=0, value=2000000)
    carbon_emission = st.number_input("Carbon Emission (tCO2)", min_value=0, value=15000000)
    transmission_capacity = st.number_input("Transmission Capacity (MW)", min_value=0, value=20000)
    transmission_completion_rate = st.slider("Transmission Completion Rate (%)", 0.0, 100.0, 70.0)
    psa_signed = st.selectbox("PSA Signed", [0, 1])
    discom_debt = st.number_input("DISCOM Debt (USD)", min_value=0, value=300000000)
    atc_loss = st.slider("AT&C Loss Percent", 0.0, 100.0, 20.0)
    land_delay = st.number_input("Land Acquisition Delay (Months)", min_value=0, value=15)
    renewable_subsidy = st.number_input("Renewable Subsidy (USD)", min_value=0, value=50000000)
    coal_subsidy = st.number_input("Coal Subsidy (USD)", min_value=0, value=200000000)
    storage_capacity = st.number_input("Energy Storage Capacity (MW)", min_value=0, value=3000)
    growth_rate = st.number_input("Economic Growth Rate (%)", min_value=0.0, value=6.5, step=0.1)
    year = st.number_input("Year", min_value=2000, value=2023)
    month = st.number_input("Month", min_value=1, max_value=12, value=6)

    submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        sample_input = {
            'Region_ID': region_id,
            'Total_Energy_Demand_MWh': total_energy_demand,
            'Coal_Power_Generation_MWh': coal_generation,
            'Carbon_Emission_tCO2': carbon_emission,
            'Transmission_Capacity_MW': transmission_capacity,
            'Transmission_Completion_Rate': transmission_completion_rate,
            'PSA_Signed': psa_signed,
            'DISCOM_Debt_USD': discom_debt,
            'AT_C_Loss_Percent': atc_loss,
            'Land_Acquisition_Delay_Months': land_delay,
            'Renewable_Subsidy_USD': renewable_subsidy,
            'Coal_Subsidy_USD': coal_subsidy,
            'Energy_Storage_Capacity_MW': storage_capacity,
            'Economic_Growth_Rate': growth_rate,
            'Year': year,
            'Month': month
        }

        try:
            prediction = predictor.predict(sample_input)
            st.success(f"🌍 Predicted Renewable Capacity: **{prediction:.2f} MW**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
