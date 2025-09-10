# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# --- Model Loading ---
# Use Streamlit's caching to load the model only once, improving performance.
@st.cache_resource
def load_predictor(model_path="pickle.pkl"):
    """
    Loads the trained model and associated preprocessing objects from a pickle file.
    
    Args:
        model_path (str): The path to the .pkl file.
        
    Returns:
        dict: A dictionary containing the model, scaler, label encoder, and feature names, or None if loading fails.
    """
    try:
        deployment_package = joblib.load(model_path)
        return deployment_package
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{model_path}'. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- Prediction Logic ---
class RenewableEnergyPredictor:
    """A class to handle the prediction process."""
    def __init__(self, deployment_package):
        """Initializes the predictor with the loaded model and preprocessors."""
        self.model = deployment_package['model']
        self.scaler = deployment_package['scaler']
        self.label_encoder = deployment_package['label_encoder']
        self.feature_names = deployment_package['feature_names']
        
    def predict(self, input_data: dict):
        """
        Preprocesses the input data and returns a prediction.
        
        Args:
            input_data (dict): A dictionary of input features from the user.
            
        Returns:
            float: The predicted renewable energy capacity.
        """
        # Convert input dictionary to a pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure the columns are in the same order as during model training
        input_df = input_df[self.feature_names]

        # Handle categorical feature: Encode Region_ID if it's a string
        if 'Region_ID' in input_df.columns and input_df['Region_ID'].dtype == 'object':
            try:
                input_df['Region_ID'] = self.label_encoder.transform(input_df['Region_ID'])
            except ValueError:
                # Handle the case where the Region_ID is not in the encoder's list of known labels
                raise ValueError(f"Region ID '{input_data['Region_ID']}' was not seen during training. Please enter a valid Region ID.")

        # Scale the numerical features using the pre-fitted scaler
        input_scaled = self.scaler.transform(input_df)
        
        # Make the prediction
        prediction = self.model.predict(input_scaled)
        
        return prediction[0]

# --- Streamlit User Interface ---
st.set_page_config(page_title="India Renewable Energy Predictor", page_icon="⚡", layout="centered")

# App title and description
st.title("⚡ India Renewable Energy Capacity Predictor")
st.write(
    "This tool predicts the potential renewable energy capacity (in MW) for a region in India. "
    "Enter the details below to get a prediction based on key infrastructure, economic, and environmental factors."
)

# Load the model and instantiate the predictor
deployment_package = load_predictor()

# Only proceed if the model loaded successfully
if deployment_package:
    predictor = RenewableEnergyPredictor(deployment_package)

    # Create a form for user input for better UX
    with st.form("prediction_form"):
        st.header("Input Features")
        
        # Organize inputs into columns for a cleaner layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Regional & Demand")
            region_id = st.text_input("Region ID (e.g., IN-MH)", "IN-MH")
            total_energy_demand = st.number_input("Total Energy Demand (MWh)", min_value=0, value=4000000, help="Total electricity demand in the region.")
            coal_generation = st.number_input("Coal Power Generation (MWh)", min_value=0, value=2000000, help="Amount of energy generated from coal sources.")
            carbon_emission = st.number_input("Carbon Emission (tCO2)", min_value=0, value=15000000, help="Total carbon dioxide emissions.")
            year = st.number_input("Year", min_value=2000, max_value=2050, value=2023)
            month = st.number_input("Month", min_value=1, max_value=12, value=6)

        with col2:
            st.subheader("Economic Factors")
            growth_rate = st.number_input("Economic Growth Rate (%)", min_value=0.0, value=6.5, step=0.1, format="%.1f")
            discom_debt = st.number_input("DISCOM Debt (USD)", min_value=0, value=300000000, help="Total debt of power distribution companies.")
            renewable_subsidy = st.number_input("Renewable Subsidy (USD)", min_value=0, value=50000000, help="Government subsidies for renewable energy projects.")
            coal_subsidy = st.number_input("Coal Subsidy (USD)", min_value=0, value=200000000, help="Government subsidies for coal power.")
            psa_signed = st.selectbox("PSA Signed", [0, 1], help="1 if Power Sale Agreement is signed, 0 otherwise.")
            
        st.divider()
        
        st.subheader("Infrastructure & Technical Factors")
        col3, col4 = st.columns(2)
        
        with col3:
            transmission_capacity = st.number_input("Transmission Capacity (MW)", min_value=0, value=20000)
            transmission_completion_rate = st.slider("Transmission Completion Rate (%)", 0.0, 100.0, 70.0, format="%.1f")
            atc_loss = st.slider("AT&C Loss Percent (%)", 0.0, 50.0, 20.0, help="Aggregate Technical & Commercial losses.", format="%.1f")
            
        with col4:
            storage_capacity = st.number_input("Energy Storage Capacity (MW)", min_value=0, value=3000, help="Capacity of energy storage systems like batteries.")
            land_delay = st.number_input("Land Acquisition Delay (Months)", min_value=0, value=15, help="Average delay in acquiring land for projects.")

        st.write("") # Add some spacing
        
        # Submit button for the form
        submitted = st.form_submit_button("🔮 Predict Capacity")

        if submitted:
            # Collect all inputs into a dictionary
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

            # Use a try-except block to handle potential errors during prediction
            try:
                # Get the prediction
                prediction = predictor.predict(sample_input)
                # Display the result
                st.success(f"### 🌍 Predicted Renewable Capacity: **{prediction:,.2f} MW**")
                st.balloons()
            except ValueError as e:
                st.error(f"Input Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")

else:
    st.warning("Application cannot start because the prediction model failed to load.")
