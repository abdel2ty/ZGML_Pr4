import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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

for col in categorical_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
for col in numeric_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)

le_dict = {}
for col in categorical_cols + ['Loan_Status']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    le_dict[col] = le

X_train = train_df.drop(columns=['Loan_ID','Loan_Status'])
y_train = train_df['Loan_Status']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

@st.cache(allow_output_mutation=True)
def train_models(X, y):
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X, y)
    svc = SVC(probability=True)
    svc.fit(X, y)
    return knn, svc

knn_model, svc_model = train_models(X_train_scaled, y_train)

status_map = {1:"Approved", 0:"Not Approved"}

# -----------------------------
# STEP 3 — Streamlit Layout
# -----------------------------
st.set_page_config(page_title="Home Loan Approval", layout="wide")
st.title("Home Loan Approval Prediction")

st.sidebar.header("About This Project")
st.sidebar.info("""
Predict whether a Home Loan will be approved using **KNN** and **SVC** classifiers.  
Fill in applicant details and get predictions in real-time.
""")

# -----------------------------
# STEP 4 — User Inputs
# -----------------------------
st.subheader("Input Features")
input_dict = {}

# Numeric inputs
input_dict['ApplicantIncome'] = st.number_input("Applicant Income", min_value=0, value=5000)
input_dict['CoapplicantIncome'] = st.number_input("Coapplicant Income", min_value=0, value=0)
input_dict['LoanAmount'] = st.number_input("Loan Amount", min_value=0, value=150)
input_dict['Loan_Amount_Term'] = st.number_input("Loan Amount Term (months)", min_value=12, value=360)
input_dict['Credit_History'] = st.selectbox("Credit History", [1.0, 0.0])

# Categorical inputs as selectbox
input_dict['Gender'] = st.selectbox("Gender", le_dict['Gender'].classes_)
input_dict['Married'] = st.selectbox("Married", le_dict['Married'].classes_)
input_dict['Dependents'] = st.selectbox("Dependents", le_dict['Dependents'].classes_)
input_dict['Education'] = st.selectbox("Education", le_dict['Education'].classes_)
input_dict['Self_Employed'] = st.selectbox("Self Employed", le_dict['Self_Employed'].classes_)
input_dict['Property_Area'] = st.selectbox("Property Area", le_dict['Property_Area'].classes_)

input_df = pd.DataFrame([input_dict])

# Encode categorical safely
for col in categorical_cols:
    le = le_dict[col]
    if input_df[col][0] in le.classes_:
        input_df[col] = le.transform(input_df[col])
    else:
        input_df[col] = le.transform([le.classes_[0]])

# Scale numeric
input_df_scaled = input_df.copy()
input_df_scaled[numeric_cols] = scaler.transform(input_df[numeric_cols])
input_df_scaled = input_df_scaled[X_train_scaled.columns].astype(float)

# -----------------------------
# STEP 5 — Predictions
# -----------------------------
st.markdown("### Predictions")  # أكبر عنوان

st.markdown("#### KNN Model")
knn_pred = knn_model.predict(input_df_scaled)[0]
knn_prob = knn_model.predict_proba(input_df_scaled)[0,1]
st.success(f"{status_map[knn_pred]} (Probability: {knn_prob:.2f})")

st.markdown("#### SVC Model")
svc_pred = svc_model.predict(input_df_scaled)[0]
svc_prob = svc_model.predict_proba(input_df_scaled)[0,1]
st.success(f"{status_map[svc_pred]} (Probability: {svc_prob:.2f})")

# -----------------------------
# STEP 6 — Show Input Data (checkbox)
# -----------------------------
if st.checkbox("Show Input Data"):
    st.dataframe(input_df)