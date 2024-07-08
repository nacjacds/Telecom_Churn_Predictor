import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pickle
import streamlit as st
from sklearn.preprocessing import OneHotEncoder


# Load your model
with open('models/modelo_gradientboosting.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title('Customer Churn Prediction App')

# Inputs for Type of Contract
contract_type = st.selectbox(
    'Type of contract',
    ['Month-to-month', 'One year', 'Two year']
)

# Mapping contract type to one-hot encoding
contract_map = {
    'Month-to-month': [1, 0, 0],
    'One year': [0, 0, 1],
    'Two year': [0, 1, 0]
}
contract_values = contract_map[contract_type]

# Input for tenure
tenure = st.slider('Tenure (in months)', min_value=0, max_value=72, step=1)

# Inputs for Services
st.h4('Hired Services')
services = {
    'InternetService_Fiber optic': st.checkbox('Fiber optic Internet Service'),
    'OnlineSecurity': st.checkbox('Online Security'),
    'OnlineBackup': st.checkbox('Online Backup'),
    'DeviceProtection': st.checkbox('Device Protection'),
    'TechSupport': st.checkbox('Tech Support'),
    'StreamingTV': st.checkbox('Streaming TV'),
    'StreamingMovies': st.checkbox('Streaming Movies'),
    'MultipleLines': st.checkbox('Multiple Lines')
}

# Inputs for Payment
payment_method = st.selectbox(
    'Payment method',
    ['Electronic check', 'Other']
)

# Mapping payment method to one-hot encoding
payment_map = {
    'Electronic check': 1,
    'Other': 0
}
payment_method_value = payment_map[payment_method]

monthly_charges = st.number_input('Monthly Charges', min_value=0.0)
total_charges = st.number_input('Total Charges', min_value=0.0)

# Inputs for Type of Customer
dependents = st.checkbox('Dependents')
partner = st.checkbox('Partner')

# Prepare the input array for the model
input_data = np.array([[
    contract_values[0], contract_values[1], contract_values[2],
    tenure,
    int(services['InternetService_Fiber optic']),
    int(services['OnlineSecurity']),
    int(services['OnlineBackup']),
    int(services['DeviceProtection']),
    int(services['TechSupport']),
    int(services['StreamingTV']),
    int(services['StreamingMovies']),
    int(services['MultipleLines']),
    payment_method_value,
    monthly_charges,
    total_charges,
    int(dependents),
    int(partner)
]])

# Button to trigger prediction
if st.button('Predict'):
    # Get the prediction from the model
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f'The predicted churn is: {"Yes" if prediction[0] == 1 else "No"}')
