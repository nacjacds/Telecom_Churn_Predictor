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

# Cargar el modelo
with open('models/gbc_pipeline_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Título de la aplicación
st.title('Customer Churn Prediction App')

# Inputs para el tipo de contrato
contract_type = st.selectbox(
    'Type of contract',
    ['Month-to-month', 'Two year']
)

# Mapeo del tipo de contrato a one-hot encoding
contract_map = {
    'Month-to-month': [1, 0],
    'Two year': [0, 1]
}
contract_values = contract_map[contract_type]

# Input para tenure
tenure = st.slider('Tenure (in months)', min_value=0, max_value=72, step=1)

# Inputs para los servicios contratados
st.write('Hired Services')
services = {
    'InternetService_Fiber optic': st.checkbox('Fiber optic Internet Service'),
    'OnlineSecurity': st.checkbox('Online Security'),
    'TechSupport': st.checkbox('Tech Support'),
    'MultipleLines': st.checkbox('Multiple Lines')
}

# Inputs para el método de pago
payment_method = st.selectbox(
    'Payment method',
    ['Electronic check', 'Other']
)

# Mapeo del método de pago a one-hot encoding
payment_map = {
    'Electronic check': 1,
    'Other': 0
}
payment_method_value = payment_map[payment_method]

monthly_charges = st.number_input('Monthly Charges', min_value=0.0)
total_charges = st.number_input('Total Charges', min_value=0.0)

# Preparar el array de entrada para el modelo
input_data = np.array([[
    contract_values[0], contract_values[1],  # Contract
    tenure,  # Tenure
    int(services['InternetService_Fiber optic']),  # InternetService_Fiber optic
    int(services['OnlineSecurity']),  # OnlineSecurity
    int(services['TechSupport']),  # TechSupport
    int(services['MultipleLines']),  # MultipleLines
    payment_method_value,  # PaymentMethod_Electronic check
    monthly_charges,  # MonthlyCharges
    total_charges  # TotalCharges
]])

# Botón para desencadenar la predicción
if st.button('Predict'):
    # Obtener la predicción del modelo
    prediction = model.predict(input_data)
    
    # Mostrar la predicción
    st.write(f'Prediction: {"The customer IS likely to churn." if prediction[0] == 1 else "The customer is NOT likely to churn."}')
