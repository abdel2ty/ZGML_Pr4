import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# -----------------------------
# STEP 1 — Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("loan_sanction_train.csv")
    test  = pd.read_csv("loan_sanction_test.csv")
    return train, test

train, test = load_data()

# Keep a copy of test IDs for submission
original_test = test.copy()

# -----------------------------
# STEP 2 — Data Cleaning
# -----------------------------
# Fill missing categorical values with mode
for col in ['Gender','Married','Dependents','Self_Employed','Credit_History','Loan_Amount_Term']:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(train[col].mode()[0], inplace=True)

# Fill missing numerical values with median
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

# -----------------------------
# STEP 3 — Feature Engineering
# -----------------------------
categorical_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
numerical_cols   = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']

# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col]  = le.transform(test[col])

# Encode target
train['Loan_Status'] = train['Loan_Status'].map({'Y':1, 'N':0})

# -----------------------------
# STEP 4 — Prepare Features & Target
# -----------------------------
features = categorical_cols + numerical_cols
target   = 'Loan_Status'

X = train[features]
y = train[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# STEP 5 — Train Models
# -----------------------------
@st.cache_resource
def train_knn_svc(X_scaled, y):
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_scaled, y)
    svc = SVC(probability=True)
    svc.fit(X_scaled, y)
    return knn, svc

knn_model, svc_model = train_knn_svc(X_scaled, y)

# -----------------------------
# STEP 6 — Streamlit Layout
# -----------------------------
# Sidebar: Project Info
st.sidebar.header("About This Project")
st.sidebar.info("""
This app predicts **Home Loan Approval** using **KNN Classifier** and **SVC** models.

- Dataset: Home Loan Sanction  
- Features: Gender, Married, Dependents, Education, Self Employed, Applicant/Coapplicant Income, Loan Amount, Loan Term, Credit History, Property Area
""")

# Main Page: Title
st.title("Home Loan Approval Prediction")

# Feature Inputs
st.subheader("Input Features")

user_inputs = {}
# Categorical Inputs
user_inputs['Gender'] = st.selectbox("Gender", ["Male","Female"])
user_inputs['Married'] = st.selectbox("Married", ["Yes","No"])
user_inputs['Dependents'] = st.selectbox("Dependents", ["0","1","2","3+"])
user_inputs['Education'] = st.selectbox("Education", ["Graduate","Not Graduate"])
user_inputs['Self_Employed'] = st.selectbox("Self Employed", ["Yes","No"])
user_inputs['Property_Area'] = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])

# Numerical Inputs
user_inputs['ApplicantIncome'] = st.number_input("Applicant Income", min_value=0, value=5000)
user_inputs['CoapplicantIncome'] = st.number_input("Coapplicant Income", min_value=0, value=1500)
user_inputs['LoanAmount'] = st.number_input("Loan Amount", min_value=0, value=120)
user_inputs['Loan_Amount_Term'] = st.number_input("Loan Amount Term (months)", min_value=12, value=360)
user_inputs['Credit_History'] = st.number_input("Credit History", min_value=0, max_value=1, value=1)

# -----------------------------
# STEP 7 — Encode & Scale User Inputs
# -----------------------------
input_df = pd.DataFrame(user_inputs, index=[0])

# Encode categorical inputs same as train
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(train[col])  # fit on train
    input_df[col] = le.transform(input_df[col])

# Scale numerical
input_scaled = scaler.transform(input_df[features])

# -----------------------------
# STEP 8 — Predict
# -----------------------------
knn_pred = knn_model.predict(input_scaled)[0]
svc_pred = svc_model.predict(input_scaled)[0]

knn_prob = knn_model.predict_proba(input_scaled)[0,1]
svc_prob = svc_model.predict_proba(input_scaled)[0,1]

# Display prediction
st.subheader("Prediction Results")
st.write(f"**KNN Prediction:** {'Approved' if knn_pred==1 else 'Rejected'} (Probability: {knn_prob:.2f})")
st.write(f"**SVC Prediction:** {'Approved' if svc_pred==1 else 'Rejected'} (Probability: {svc_prob:.2f})")