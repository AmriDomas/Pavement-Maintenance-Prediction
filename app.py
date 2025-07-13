import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=FutureWarning)

# Load the dataset (This part should ideally be done once when the app starts)
# Using st.cache_resource to load data only once
@st.cache_resource
def load_data():
    url = "https://huggingface.co/datasets/11amri/Pavement/resolve/main/ESC%2012%20Pavement%20Dataset.csv"
    return pd.read_csv(url)

# Load the model (This part should ideally be done once when the app starts)
@st.cache_resource
def load_model(x_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(x_train, y_train)
    return model

# Feature Engineering function
def engineer_features(df):
    df['Years Since Maintenance'] = 2025 - df['Last Maintenance']
    df['Is Old Road'] = df['Years Since Maintenance'] > 10

    df['AADT'] = df['AADT'].apply(lambda x: x if x >= 0 else np.nan)
    df['AADT'].fillna(df['AADT'].median(), inplace=True) # Fill NaN with median AFTER engineering

    df['Traffic Level'] = pd.cut(df['AADT'],
                          bins=[-0.1, 5000, 15000, 1e9],
                          labels=['Low', 'Medium', 'High'],
                          right=True)

    def categorize_pci(pci):
        if pci <= 40:
            return 'Poor'
        elif pci <= 70:
            return 'Fair'
        else:
            return 'Good'
    df['PCI_Class'] = df['PCI'].apply(categorize_pci)

    def categorize_rainfall(rainfall):
        if rainfall < 50:
            return 'Low'
        elif 50 <= rainfall <= 75:
            return 'Medium'
        else:
            return 'High'
    df['Rainfall Category'] = df['Average Rainfall'].apply(categorize_rainfall)

    df['Age_Rain_Interaction'] = df['Years Since Maintenance'] * df['Average Rainfall']
    df['Age_Rutting_Interaction'] = df['Years Since Maintenance'] * df['Rutting']
    df['Age_IRI_Interaction'] = df['Years Since Maintenance'] * df['IRI']
    df['Age_PCI_Interaction'] = df['Years Since Maintenance'] * df['PCI']

    df['Severity_Factor'] = (100 - df['PCI']) / 100

    # Handle potential NaN in Rutting and IRI before interaction/cost calculation
    df['Rutting'] = df['Rutting'].fillna(0) # Or use a more appropriate imputation
    df['IRI'] = df['IRI'].fillna(0) # Or use a more appropriate imputation

    # Road length simulation - need a deterministic way for prediction
    # For prediction, we might need to get this from input or have a lookup
    # For simplicity in the app, let's make it an input or a fixed value for demonstration
    # Assuming a fixed length for prediction input
    if 'Road_Length_km' not in df.columns:
        df['Road_Length_km'] = 50 # Example fixed length for new predictions

    cost_map = {
        ('Primary', 'Asphalt'): 70000,
        ('Primary', 'Concrete'): 90000,
        ('Secondary', 'Asphalt'): 50000,
        ('Secondary', 'Concrete'): 70000,
        ('Tertiary', 'Asphalt'): 30000,
        ('Tertiary', 'Concrete'): 45000
    }

    df['Road Type'] = df['Road Type'].str.strip()
    df['Asphalt Type'] = df['Asphalt Type'].str.strip()

    df['Road_Asphalt_Key'] = list(zip(df['Road Type'], df['Asphalt Type']))

    df['Cost_per_km'] = df['Road_Asphalt_Key'].map(cost_map).fillna(40000) # Handle potential keys not in map
    df['Estimated_Cost'] = (
        df['Road_Length_km'] *
        df['Cost_per_km']
    )

    df['Cost_Efficiency'] = np.where(
        (df['Severity_Factor'] == 0) | (df['Road_Length_km'].clip(lower=0.01) == 0), 0, # Avoid division by zero
        df['Estimated_Cost'] / (
            df['Road_Length_km'].clip(lower=0.01) *
            df['Severity_Factor'].clip(lower=0.01) *
            (1 + df['Rutting'].fillna(0)/10) * # Handle potential NaN in Rutting/IRI
            (1 + df['IRI'].fillna(0)/2)
        )
    )

    # Store original categorical columns before one-hot and ordinal encoding
    original_road_type = df['Road Type']
    original_asphalt_type = df['Asphalt Type']

    # One-hot encode
    df = pd.get_dummies(df, columns=['Road Type', 'Asphalt Type'], drop_first=True)

    # Ordinal encode
    ordinal_cols_map = {
        'Traffic Level': ['Low', 'Medium', 'High'],
        'PCI_Class': ['Poor', 'Fair', 'Good'],
        'Rainfall Category': ['Low', 'Medium', 'High']
    }
    encoder = OrdinalEncoder(categories=[ordinal_cols_map[col] for col in ordinal_cols_map])
    # Need to ensure columns exist before encoding
    cols_to_encode = [col for col in ordinal_cols_map.keys() if col in df.columns]
    if cols_to_encode:
         df[cols_to_encode] = encoder.fit_transform(df[cols_to_encode])


    # Restore original categorical columns for display/cost calculation later if needed
    df['Original Road Type'] = original_road_type
    df['Original Asphalt Type'] = original_asphalt_type

    return df


# Prepare data for training the model
# This will only run once due to st.cache_resource
@st.cache_resource
def prepare_training_data():
    pavement_train = load_data()
    # Handle negative values before feature engineering that relies on them
    pavement_train['AADT'] = pavement_train['AADT'].apply(lambda x: x if x >= 0 else np.nan)
    pavement_train['IRI'] = pavement_train['IRI'].apply(lambda x: x if x >= 0 else np.nan) # Assuming IRI shouldn't be negative

    # Impute NaNs BEFORE engineering features that use these columns
    imputer_median = SimpleImputer(strategy='median')
    pavement_train[['AADT', 'IRI']] = imputer_median.fit_transform(pavement_train[['AADT', 'IRI']])
    # Also handle any potential NaNs in other columns used in FE
    imputer_zero = SimpleImputer(strategy='constant', fill_value=0)
    pavement_train[['Rutting', 'Average Rainfall', 'PCI', 'Last Maintenance']] = imputer_zero.fit_transform(pavement_train[['Rutting', 'Average Rainfall', 'PCI', 'Last Maintenance']])


    pavement_engineered = engineer_features(pavement_train.copy())

    x = pavement_engineered.drop(['Needs Maintenance', 'Segment ID','Road_Asphalt_Key', 'Original Road Type', 'Original Asphalt Type'], axis=1, errors='ignore')
    y = pavement_engineered['Needs Maintenance']

    # Align columns between training data and potentially new prediction data
    # This is crucial for the model to work correctly on new input
    # We will store the training columns to reindex prediction data later
    training_cols = x.columns
    return x, y, training_cols


# Load data and train model
x_train, y_train, training_cols = prepare_training_data()
model = load_model(x_train, y_train)


# Streamlit App Title
st.title("Pavement Maintenance Prediction App")
st.write("Predict if a road segment needs urgent maintenance and analyze potential repair costs.")

# --- Input Section ---
st.header("Enter Road Segment Details")

col1, col2 = st.columns(2)

with col1:
    segment_id_input = st.text_input("Segment ID", "NewSegment123")
    pci_input = st.slider("Pavement Condition Index (PCI)", 0, 100, 60)
    aadt_input = st.number_input("Average Annual Daily Traffic (AADT)", min_value=0, value=10000)
    rainfall_input = st.number_input("Average Annual Rainfall (mm)", min_value=0.0, value=70.0)
    road_length_input = st.number_input("Road Length (km)", min_value=0.1, value=10.0)

with col2:
    road_type_input = st.selectbox("Road Type", ['Primary', 'Secondary', 'Tertiary'])
    asphalt_type_input = st.selectbox("Asphalt Type", ['Asphalt', 'Concrete'])
    last_maintenance_input = st.number_input("Year of Last Maintenance", min_value=1900, value=2015, max_value=2025)
    rutting_input = st.number_input("Rutting Depth (mm)", min_value=0.0, value=5.0)
    iri_input = st.number_input("International Roughness Index (IRI) (m/km)", min_value=0.0, value=3.0)


# Create a DataFrame from the user input
input_data = pd.DataFrame({
    'Segment ID': [segment_id_input],
    'PCI': [pci_input],
    'Road Type': [road_type_input],
    'AADT': [aadt_input],
    'Asphalt Type': [asphalt_type_input],
    'Last Maintenance': [last_maintenance_input],
    'Average Rainfall': [rainfall_input],
    'Rutting': [rutting_input],
    'IRI': [iri_input],
    'Road_Length_km': [road_length_input] # Include the input road length
})

# --- Prediction Section ---
st.header("Prediction Results")

if st.button("Predict Maintenance Need"):
    # Apply feature engineering to the input data
    # Need to ensure input columns match the columns expected by engineer_features
    # Engineer features on a copy to avoid modifying the original input_data
    input_data_engineered = engineer_features(input_data.copy())

    # Select only the features used for training
    # Need to ensure input columns match training columns BEFORE predicting
    # Add missing columns with default values (e.g., 0 for one-hot encoded)
    # and reindex to match the training column order
    input_features = input_data_engineered.drop(['Segment ID', 'Needs Maintenance', 'Road_Asphalt_Key', 'Original Road Type', 'Original Asphalt Type'], axis=1, errors='ignore')

    # Add missing columns present in training_cols but not in input_features
    missing_cols = set(training_cols) - set(input_features.columns)
    for c in missing_cols:
        input_features[c] = 0 # Assuming 0 is a safe default for engineered features/one-hot

    # Ensure the order of columns is the same as in training data
    input_features = input_features[training_cols]

    # Make prediction
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)[:, 1]

    # Display prediction
    st.subheader("Maintenance Prediction")
    if prediction[0] == 1:
        st.error(f"Prediction: **URGENT MAINTENANCE NEEDED** (Probability: {prediction_proba[0]:.2f})")
    else:
        st.success(f"Prediction: **No Urgent Maintenance Needed** (Probability: {prediction_proba[0]:.2f})")

    # --- Cost Analysis Section (only if maintenance is needed) ---
    if prediction[0] == 1:
        st.header("Estimated Cost Analysis")
        estimated_cost = input_data_engineered['Estimated_Cost'].iloc[0]
        severity_factor = input_data_engineered['Severity_Factor'].iloc[0]
        cost_efficiency = input_data_engineered['Cost_Efficiency'].iloc[0]

        st.write(f"Based on Road Type ({input_data_engineered['Original Road Type'].iloc[0]}) and Asphalt Type ({input_data_engineered['Original Asphalt Type'].iloc[0]}), and Road Length ({road_length_input:.2f} km):")
        st.write(f"Estimated Repair Cost: **${estimated_cost:,.2f}**")
        st.write(f"Severity Factor: **{severity_factor:.2f}**")
        st.write(f"Cost Efficiency Score (Lower is better): **{cost_efficiency:,.2f}**")

        st.subheader("Potential ROI Simulation (if repaired)")

        # Simulate ROI for this single segment
        damage_cost_per_km_per_year = st.number_input("Assumed yearly damage cost per km if not repaired ($)", value=18000, min_value=0)
        life_extension_years = st.number_input("Assumed years of life extension after repair", value=5, min_value=1)

        avoided_cost = (
            road_length_input *
            damage_cost_per_km_per_year *
            life_extension_years *
            severity_factor # Use the calculated severity
        )

        if estimated_cost > 0: # Avoid division by zero
             roi = (avoided_cost - estimated_cost) / estimated_cost
             st.write(f"Potential Avoided Cost over {life_extension_years} years: **${avoided_cost:,.2f}**")
             st.write(f"Potential Return on Investment (ROI): **{roi:.2f}**")
        else:
             st.write("Estimated cost is zero, cannot calculate ROI.")


        st.subheader("Impact of Repair Delay Simulation")
        max_delay_years = st.slider("Simulate delay up to (years)", 1, 10, 5)
        damage_degradation_per_year = st.slider("Assumed PCI degradation per year of delay", 1, 10, 5)
        cost_increase_rate = st.slider("Assumed annual cost increase rate due to delay", 0.01, 0.2, 0.1, step=0.01)

        # Simulate delay for this single segment
        delay_results = []
        for delay in range(1, max_delay_years + 1):
             delayed_pci = max(0, pci_input - delay * damage_degradation_per_year)
             delayed_severity = (100 - delayed_pci) / 100
             delayed_cost_per_km = input_data_engineered['Cost_per_km'].iloc[0] * ((1 + cost_increase_rate) ** delay)
             delayed_estimated_cost = road_length_input * delayed_cost_per_km * delayed_severity

             # Recalculate avoided cost based on *delayed* severity for consistency in simulation
             delayed_avoided_cost = (
                road_length_input *
                damage_cost_per_km_per_year *
                life_extension_years *
                delayed_severity
             )

             if delayed_estimated_cost > 0:
                 roi_after_delay = (delayed_avoided_cost - delayed_estimated_cost) / delayed_estimated_cost
             else:
                 roi_after_delay = -np.inf # Represent infinite loss if cost is zero but damage is high

             delay_results.append({
                 'Years_Delayed': delay,
                 'Delayed_PCI': delayed_pci,
                 'Estimated_Cost': delayed_estimated_cost,
                 'ROI_after_delay': roi_after_delay
             })

        delay_df_single = pd.DataFrame(delay_results)

        st.dataframe(delay_df_single)

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax1 = plt.subplots(figsize=(10, 6))

        sns.lineplot(data=delay_df_single, x='Years_Delayed', y='Estimated_Cost', marker='o', label='Estimated Repair Cost', ax=ax1, color='blue')
        ax1.set_xlabel("Years Delayed")
        ax1.set_ylabel("Estimated Repair Cost ($)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        sns.lineplot(data=delay_df_single, x='Years_Delayed', y='ROI_after_delay', marker='s', label='Potential ROI', ax=ax2, color='green')
        ax2.set_ylabel("Potential ROI", color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        fig.suptitle("Impact of Repair Delay on Cost and Potential ROI", fontsize=16)
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        st.pyplot(fig)

    else:
        st.info("This segment is predicted to NOT need urgent maintenance at this time.")

# --- About Section ---
st.sidebar.header("About")
st.sidebar.write("""
This app uses a trained XGBoost model to predict whether a road segment requires urgent maintenance based on various factors like PCI, AADT, Rutting, IRI, Last Maintenance, etc.
It also provides an estimated repair cost and potential ROI analysis for segments identified as needing maintenance.
""")
st.sidebar.header("Disclaimer")
st.sidebar.write("""
This is a demonstration based on the provided dataset and model. The cost estimations and ROI calculations are based on assumptions and are for illustrative purposes only. Actual costs and benefits may vary.
""")

# Ensure necessary libraries are installed (for Colab/Jupyter)
# In a real Streamlit app deployment, you would use requirements.txt
# !pip install -q streamlit xgboost sklearn pandas numpy matplotlib seaborn kagglehub