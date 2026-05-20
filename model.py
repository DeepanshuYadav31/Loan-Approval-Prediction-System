import streamlit as st
import pandas as pd
import numpy as np
from math import log


def predict_loan_status(data):
    
    # Map Categorical Features
    self_employed = 1 if data['self_employed'] == 'Yes' else 0
    
    # 2. Extract and Process Numerical Features
    cibil_score = data['cibil_score']
    income_annum = data['income_annum']
    loan_amount = data['loan_amount']
    
    total_assets = (
        data['residential_assets_value'] +
        data['commercial_assets_value'] +
        data['luxury_assets_value'] +
        data['bank_asset_value']
    )
    
    # Log Transformation for highly skewed income/loan/asset features (Avoids log(0))
    income_log = log(income_annum + 1)
    loan_log = log(loan_amount + 1)
    assets_log = log(total_assets + 1)
    
    # --- 3. Calculate a 'Prediction Score' (Heuristic) ---
    score = 0
    
    # Weight 1: CIBIL Score (Creditworthiness - Most Critical)
    # 900 (perfect) -> high positive, 300 (poor) -> high negative
    score += (cibil_score - 300) * 0.15 # Max 90 points (e.g., 600 * 0.15)
    
    # Weight 2: Income vs. Loan Amount (High Importance)
    # Higher income_log (positive) - Higher loan_log (negative)
    score += income_log * 4 
    score -= loan_log * 3
    
    # Weight 3: Total Assets (Collateral & Net Worth)
    score += assets_log * 2
    
    # Weight 4: Self-Employed (Slight Risk Indicator)
    if self_employed == 1:
        score -= 5 # Minor penalty for slightly higher income volatility
        
    # Weight 5: Loan Term (Slight Influence - longer term means smaller payments, but higher risk)
    # Penalty for very long terms (480 months) and very short terms (12 months)
    loan_term = data['loan_term']
    if loan_term > 360 or loan_term < 60:
        score -= 2
    
    # --- 4. Determine Final Status based on Score Thresholds ---
    # The thresholds are adjusted for the new scoring system.
    if score >= 90:
        return 'Y' # Approved
    else:
        return 'N' # Rejected

# --- 2. Streamlit UI and Input Handling ---

st.set_page_config(
    page_title="Loan Approval Predictor (RF Model)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💰 Loan Approval Prediction System")
st.subheader("Predictive Model using Applicant Financial Data and Assets")

# --- Sidebar for User Inputs ---
st.sidebar.header("Applicant Financial Profile")
st.sidebar.markdown("---")


# CIBIL Score - The new crucial feature
st.sidebar.subheader("Credit and Employment")
cibil_score = st.sidebar.number_input(
    "CIBIL Score (300-900)", 
    value=750, 
    min_value=300, 
    max_value=900, 
    step=1
)
self_employed = st.sidebar.selectbox("Self Employed", ('Yes', 'No'), index=1)

st.sidebar.markdown("---")
st.sidebar.header("Income and Loan Details (in USD)")

# Income and Loan
income_annum = st.sidebar.number_input("Annual Income", value=75000, min_value=0, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount Requested", value=200000, min_value=1000, step=5000)
loan_term = st.sidebar.number_input("Loan Term (Months)", value=360, min_value=1, max_value=480, step=12)


st.sidebar.markdown("---")
st.sidebar.header("Asset Values (Collateral)")

# Asset Values
residential_assets_value = st.sidebar.number_input("Residential Assets Value", value=150000, min_value=0, step=10000)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", value=0, min_value=0, step=10000)
luxury_assets_value = st.sidebar.number_input("Luxury Assets Value", value=25000, min_value=0, step=5000)
bank_asset_value = st.sidebar.number_input("Bank Asset Value (Savings/Investments)", value=15000, min_value=0, step=1000)


# --- 3. Prediction Execution ---

# Collect the exact user data columns requested
user_data = {
    'self_employed': self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

st.markdown("---")

if st.button('Predict Loan Status', help="Click to run the prediction simulation based on your Random Forest model features."):
    # Run the prediction
    prediction = predict_loan_status(user_data)
    
    # --- 4. Display Results ---
    st.header("Prediction Result")
    
    if prediction == 'Y':
        st.success("✅ Prediction: Loan **APPROVED**")
        st.balloons()
        st.markdown(
            """
            <div style="padding: 15px; border-radius: 10px; border: 2px solid #38a169; background-color: #f0fff4; color: black;">
                <p><strong>The applicant demonstrates strong creditworthiness and financial stability.</strong></p>
                <ul>
                    <li>The CIBIL Score indicates a good payment history.</li>
                    <li>The combined assets provide substantial collateral.</li>
                    <li>The income supports the requested loan amount.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.error("❌ Prediction: Loan **REJECTED**")
        st.markdown(
            """
            <div style="padding: 15px; border-radius: 10px; border: 2px solid #e53e3e; background-color: #fff5f5; color: black;">
                <p><strong>The application does not meet the minimum criteria for approval.</strong></p>
                <p>Common contributing factors for rejection often include:</p>
                <ul>
                    <li>Low CIBIL Score (below 650-700).</li>
                    <li>Insufficient Annual Income relative to the Loan Amount.</li>
                    <li>Low collateral (total asset value).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Display the input data used for the prediction
    st.caption("Input Data Used for Prediction:")
    # Create a cleaner dictionary for JSON display
    display_data = {key.replace('_', ' ').title(): value for key, value in user_data.items()}
    st.json(display_data)

st.sidebar.markdown("---")
st.sidebar.caption("App created using Streamlit and a simulated Random Forest model.")