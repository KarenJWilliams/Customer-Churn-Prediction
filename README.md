![alt text](https://github.com/KarenJWilliams/Customer-Churn-Prediction/blob/main/Website%20Image%201.jpg)?raw=true)

# 📉 Customer Churn Prediction

A *Data Science Capstone Project* built using Python, Machine Learning, and Streamlit to predict customer churn based on demographic and service-related data.  
The project includes *data preprocessing, exploratory data analysis (EDA), model building, evaluation, and deployment* via an interactive Streamlit dashboard.

---

## 🧠 Objective

The main goal of this project is to predict whether a customer is likely to *churn (discontinue a service)* or remain with the company based on their historical data.  
By identifying high-risk customers, businesses can take proactive steps to improve customer retention and satisfaction.

---

## 📂 Dataset Information

The dataset used in this project was obtained from [Kaggle - Customer Churn Dataset](https://www.kaggle.com/datasets/rashadrmammadov/customer-churn-dataset).  
It contains various customer-related attributes such as:

- *Demographic Information:* Gender, Senior Citizen, Partner, Dependents  
- *Account Information:* Tenure, Contract type, Payment method, Monthly and Total charges  
- *Services Subscribed:* Internet service type, Online security, Tech support, Streaming services, etc.

---

## 🧩 Project Workflow

### *Phase 1: Data Loading and Exploration*
- Loaded dataset using pandas.
- Displayed sample records, data types, and checked for missing values.
- Explored numerical and categorical columns.
- Used visualizations (bar plots, histograms, heatmaps) to understand data distribution and relationships.

### *Phase 2: Data Cleaning and Preprocessing*
- Handled missing and inconsistent values.
- Converted TotalCharges to numeric type.
- Encoded categorical variables using LabelEncoder and OneHotEncoder.
- Standardized numerical features using StandardScaler.
- Split the dataset into training and testing sets (80/20 ratio).

### *Phase 3: Exploratory Data Analysis (EDA)*
- Analyzed churn rate by contract type, internet service, and payment method.
- Identified key churn indicators:
  - *Month-to-month contracts* had the highest churn rate.
  - *Electronic check* users showed higher churn tendencies.
  - Customers with *no tech support or online security* were more likely to churn.

### *Phase 4: Model Building*
Trained multiple machine learning models and compared their performance:
- *Logistic Regression*
- *Random Forest Classifier*
- *XGBoost Classifier*

*Best Model:* Random Forest achieved the highest accuracy and balanced performance between precision and recall.

*Why Random Forest?*
- Handles both categorical and numerical data efficiently.
- Robust against overfitting.
- Provides feature importance insights.

### *Phase 5: Feature Selection*
Selected key features based on correlation analysis and model importance:
- tenure
- MonthlyCharges
- TotalCharges
- Contract
- InternetService
- OnlineSecurity
- TechSupport
- PaymentMethod
- SeniorCitizen

These features directly influence customer satisfaction, billing stability, and service dependency—key factors in predicting churn.

### *Phase 6: Model Evaluation*
- Evaluated models using:
  - *Accuracy*
  - *Precision*
  - *Recall*
  - *F1 Score*
  - *ROC-AUC Curve*
- Visualized confusion matrix and feature importance chart.

### *Phase 7: Model Deployment*
Deployed the final model using *Streamlit* for an interactive web dashboard.

---

## 🎛 Streamlit Dashboard

### *Features*
- Accepts user input for key features.
- Predicts whether the customer will churn (Yes or No).
- Displays churn probability score.
- Simple and user-friendly interface.

### *How to Run the Dashboard*
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
