# ============================================================
# app.py — Retail Sales Profit Prediction Web App
# Streamlit frontend for the Random Forest model
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Profit Predictor",
    page_icon="🛒",
    layout="centered"
)

# ── Load model artifact ──────────────────────────────────────
@st.cache_resource
def load_artifact():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

if not os.path.exists("model.pkl"):
    st.error("⚠️ model.pkl not found. Please run `python train_and_save_model.py` first.")
    st.stop()

artifact = load_artifact()
model         = artifact['model']
encoders      = artifact['encoders']
features      = artifact['features']
categories    = artifact['categories']
sub_cat_map   = artifact['sub_categories']
regions       = artifact['regions']
segments      = artifact['segments']
ship_modes    = artifact['ship_modes']
r2            = artifact['r2_score']

# ── Header ────────────────────────────────────────────────────
st.title("🛒 Automated Retail Sales Profit Predictor")
st.markdown(
    "Predict the **profit** of a retail transaction using a "
    "**Random Forest** model trained on Superstore-style sales data."
)
st.success(f"✅ Model Accuracy (R² Score): **{r2:.4f}** ({r2*100:.2f}% variance explained)")

st.divider()

# ── Input Form ────────────────────────────────────────────────
st.subheader("📋 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Category", categories)
    sub_category = st.selectbox("Sub-Category", sub_cat_map[category])
    region = st.selectbox("Region", regions)
    segment = st.selectbox("Segment", segments)

with col2:
    ship_mode = st.selectbox("Ship Mode", ship_modes)
    sales = st.number_input("Sales Amount ($)", min_value=1.0, max_value=10000.0,
                             value=500.0, step=10.0)
    quantity = st.number_input("Quantity", min_value=1, max_value=50, value=3, step=1)
    discount = st.slider("Discount Rate", min_value=0.0, max_value=0.5, value=0.0, step=0.05)

col3, col4 = st.columns(2)
with col3:
    year = st.selectbox("Year", [2023, 2024, 2025, 2026], index=1)
with col4:
    month = st.selectbox("Month", list(range(1, 13)), index=5)

st.divider()

# ── Prediction ────────────────────────────────────────────────
if st.button("🔮 Predict Profit", type="primary", use_container_width=True):

    input_data = pd.DataFrame([{
        'Sales': sales,
        'Quantity': quantity,
        'Discount': discount,
        'Year': year,
        'Month': month,
        'Category_enc': encoders['Category'].transform([category])[0],
        'Sub_Category_enc': encoders['Sub_Category'].transform([sub_category])[0],
        'Region_enc': encoders['Region'].transform([region])[0],
        'Segment_enc': encoders['Segment'].transform([segment])[0],
        'Ship_Mode_enc': encoders['Ship_Mode'].transform([ship_mode])[0],
    }])[features]

    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")

    if prediction >= 0:
        st.success(f"### 💰 Predicted Profit: **${prediction:,.2f}**")
        margin = (prediction / sales) * 100
        st.metric("Profit Margin", f"{margin:.2f}%")
    else:
        st.error(f"### ⚠️ Predicted Loss: **${prediction:,.2f}**")
        st.warning("This transaction is predicted to result in a loss. "
                   "Consider reducing the discount rate.")

    # Quick breakdown
    with st.expander("📄 Transaction Summary"):
        summary = pd.DataFrame({
            "Field": ["Category", "Sub-Category", "Region", "Segment", "Ship Mode",
                      "Sales", "Quantity", "Discount", "Predicted Profit"],
            "Value": [category, sub_category, region, segment, ship_mode,
                      f"${sales:,.2f}", str(quantity), f"{discount:.0%}",
                      f"${prediction:,.2f}"]
        })
        st.table(summary)

    # Discount warning
    if discount >= 0.3:
        st.info("💡 **Insight:** Transactions with discount ≥ 30% earn on average "
                "62% less profit than those with lower discounts.")

st.divider()
st.caption("Built with Streamlit | Random Forest Regression | "
           "Automated Retail Sales Analysis & Profit Prediction System")
