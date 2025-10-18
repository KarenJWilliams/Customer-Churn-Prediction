import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# =============================
# 🎯 PAGE CONFIGURATION
# =============================
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide"
)

# =============================
# 🧠 LOAD TRAINED MODEL
# =============================
@st.cache_resource
def load_model():
    model = joblib.load('final_churn_model.pkl')
    return model

model = load_model()

# =============================
# 🎨 PAGE HEADER
# =============================
st.title("📉 Customer Churn Prediction Dashboard")
st.write("This dashboard predicts whether a customer is likely to churn based on their service details.")

# =============================
# 🧾 INPUT SECTION
# =============================
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input('Tenure (Months)', 0, 100, 12)
    senior = st.selectbox('Senior Citizen', [0, 1])
    contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])

with col2:
    internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No'])

with col3:
    monthly = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
    total = st.number_input('Total Charges', 0.0, 8000.0, 1500.0)
    payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])

# =============================
# 🔍 PREDICTION SECTION
# =============================
if st.button("🔮 Predict Churn"):
    # Prepare input dataframe
    input_df = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'Contract': contract,
        'InternetService': internet,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'PaymentMethod': payment,
        'SeniorCitizen': senior
    }])

    # Run prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Show results
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    st.write(f"**Predicted Churn:** {'✅ No' if pred == 0 else '⚠️ Yes'}")
    st.write(f"**Probability of Churn:** {prob:.2%}")

    if pred == 1:
        st.warning("The customer is likely to churn. Consider offering discounts or improved services.")
    else:
        st.success("The customer is likely to stay loyal.")

# =============================
# 📈 OPTIONAL VISUALIZATION SECTION
# =============================
st.markdown("---")
st.subheader("📉 Data Insights (Optional)")

uploaded_file = st.file_uploader("Upload Dataset (Optional) to Visualize Insights", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Contract Type vs Churn")
        fig1 = px.histogram(df, x='Contract', color='Churn', barmode='group')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.write("### Internet Service vs Churn")
        fig2 = px.histogram(df, x='InternetService', color='Churn', barmode='group')
        st.plotly_chart(fig2, use_container_width=True)
