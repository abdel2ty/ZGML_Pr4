import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# -----------------------------
# STEP 1 — Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("loan_sanction_train.csv")
    test = pd.read_csv("loan_sanction_test.csv")
    return train, test

train_df, test_df = load_data()

# -----------------------------
# STEP 2 — Data Cleaning
# -----------------------------
categorical_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
numeric_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']

# Fill missing categorical with mode
for col in categorical_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)

# Fill missing numeric with median
for col in numeric_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)

# Encode categorical features
le_dict = {}
for col in categorical_cols + ['Loan_Status']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    le_dict[col] = le

# Features & target
X_train = train_df.drop(columns=['Loan_ID','Loan_Status'])
y_train = train_df['Loan_Status']

# Scale numeric features only
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

# -----------------------------
# STEP 3 — Train Models
# -----------------------------
@st.cache(allow_output_mutation=True)
def train_models(X, y):
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X, y)
    
    svc = SVC(probability=True)
    svc.fit(X, y)
    
    return knn, svc

knn_model, svc_model = train_models(X_train_scaled, y_train)

# -----------------------------
# STEP 4 — Streamlit Layout
# -----------------------------
st.title("Home Loan Approval Prediction")

st.sidebar.header("About This Project")
st.sidebar.info("""
This app predicts whether a **Home Loan will be Approved or Not**.  
Models used: **KNN Classifier** & **SVC Classifier**.  

- Fill in the applicant's details below  
- Prediction is shown in real-time
""")

# -----------------------------
# STEP 5 — User Inputs
# -----------------------------
st.subheader("Applicant Details")
input_dict = {}

# Numeric inputs
input_dict['ApplicantIncome'] = st.number_input("Applicant Income", min_value=0, value=5000)
input_dict['CoapplicantIncome'] = st.number_input("Coapplicant Income", min_value=0, value=0)
input_dict['LoanAmount'] = st.number_input("Loan Amount", min_value=0, value=150)
input_dict['Loan_Amount_Term'] = st.number_input("Loan Amount Term (in months)", min_value=12, value=360)
input_dict['Credit_History'] = st.selectbox("Credit History", [1.0, 0.0])

# Categorical inputs (use original labels)
input_dict['Gender'] = st.selectbox("Gender", train_df['Gender'].unique())
input_dict['Married'] = st.selectbox("Married", train_df['Married'].unique())
input_dict['Dependents'] = st.selectbox("Dependents", train_df['Dependents'].unique())
input_dict['Education'] = st.selectbox("Education", train_df['Education'].unique())
input_dict['Self_Employed'] = st.selectbox("Self Employed", train_df['Self_Employed'].unique())
input_dict['Property_Area'] = st.selectbox("Property Area", train_df['Property_Area'].unique())

input_df = pd.DataFrame([input_dict])

# Encode categorical safely
for col in categorical_cols:
    le = le_dict[col]
    if input_df[col][0] in le.classes_:
        input_df[col] = le.transform(input_df[col])
    else:
        input_df[col] = le.transform([le.classes_[0]])  # use mode if new value

# Scale numeric
input_df_scaled = input_df.copy()
input_df_scaled[numeric_cols] = scaler.transform(input_df[numeric_cols])

# -----------------------------
# STEP 6 — Prediction
# -----------------------------
st.subheader("Predictions")
knn_pred = knn_model.predict(input_df_scaled)[0]
knn_prob = knn_model.predict_proba(input_df_scaled)[0,1]

svc_pred = svc_model.predict(input_df_scaled)[0]
svc_prob = svc_model.predict_proba(input_df_scaled)[0,1]

status_map = {1:"Approved", 0:"Not Approved"}

st.write(f"**KNN Prediction:** {status_map[knn_pred]} (Probability: {knn_prob:.2f})")
st.write(f"**SVC Prediction:** {status_map[svc_pred]} (Probability: {svc_prob:.2f})")

# Optional: Show raw input data
if st.checkbox("Show Input Data"):
    st.dataframe(input_df)