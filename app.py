import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="India Renewable Energy Dashboard", layout="wide")

# Title and description
st.title("India Renewable Energy Dashboard")
st.markdown("""
This dashboard provides an interactive exploration of the India Renewable Energy dataset, 
including data overview, visualizations, and machine learning model performance.
""")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('India_Renewable_Energy.csv')
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Visualizations", "Model Performance"])

# Data Overview Page
if page == "Data Overview":
    st.header("Data Overview")
    
    st.subheader("Dataset Preview")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())
    
    st.write("Last 5 rows of the dataset:")
    st.dataframe(df.tail())
    
    st.subheader("Dataset Shape")
    st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    st.subheader("Dataset Info")
    buffer = pd.DataFrame(df.dtypes, columns=['Data Type'])
    buffer['Missing Values'] = df.isnull().sum()
    buffer['Unique Values'] = df.nunique()
    st.dataframe(buffer)

# Visualizations Page
elif page == "Visualizations":
    st.header("Data Visualizations")
    
    # Filter by Region_ID
    regions = df['Region_ID'].unique()
    selected_region = st.selectbox("Select Region", ['All'] + list(regions))
    
    # Filter by Time Period
    time_periods = df['Time_Period'].unique()
    selected_time = st.selectbox("Select Time Period", ['All'] + list(time_periods))
    
    # Apply filters
    filtered_df = df
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region_ID'] == selected_region]
    if selected_time != 'All':
        filtered_df = filtered_df[filtered_df['Time_Period'] == selected_time]
    
    st.subheader("Filtered Data")
    st.dataframe(filtered_df)
    
    # Plot Renewable Capacity Added
    st.subheader("Renewable Capacity Added Over Time")
    if not filtered_df.empty:
        fig = px.line(filtered_df, x='Time_Period', y='Renewable_Capacity_Added_MW', 
                      color='Region_ID', title="Renewable Capacity Added (MW)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")
    
    # Plot Carbon Emissions
    st.subheader("Carbon Emissions Distribution")
    fig = px.histogram(filtered_df, x='Carbon_Emission_tCO2', 
                       title="Distribution of Carbon Emissions (tCO2)")
    st.plotly_chart(fig, use_container_width=True)

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance")
    
    st.subheader("Random Forest Regressor")
    st.markdown("""
    The Random Forest Regressor was identified as the best model with the lowest Mean Squared Error (MSE).
    Below are the steps and results of the model training and evaluation.
    """)
    
    # Prepare data for modeling
    features = ['Total_Energy_Demand_MWh', 'Coal_Power_Generation_MWh', 
                'Transmission_Capacity_MW', 'Transmission_Completion_Rate', 
                'DISCOM_Debt_USD', 'AT_C_Loss_Percent', 
                'Land_Acquisition_Delay_Months', 'Renewable_Subsidy_USD', 
                'Coal_Subsidy_USD', 'Energy_Storage_Capacity_MW', 
                'Economic_Growth_Rate']
    target = 'Renewable_Capacity_Added_MW'
    
    # Drop rows with missing values in features or target
    model_df = df[features + [target]].dropna()
    X = model_df[features]
    y = model_df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', 
                 title="Feature Importance in Random Forest Model")
    st.plotly_chart(fig, use_container_width=True)   