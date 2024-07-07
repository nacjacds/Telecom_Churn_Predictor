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
from sklearn.compose import ColumnTransformer

# Cargar el modelo
try:
    with open('/mnt/data/modelo_gradientboosting.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except FileNotFoundError:
    st.error("El archivo del modelo no se encuentra en la ruta especificada.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Definir las columnas categóricas y numéricas
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Crear el transformador de columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough'  # Deja las columnas numéricas sin cambios
)

# Crear un DataFrame de ejemplo para ajustar el codificador
dummy_data = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No phone service'],
    'InternetService': ['DSL'],
    'OnlineSecurity': ['No internet service'],
    'OnlineBackup': ['No internet service'],
    'DeviceProtection': ['No internet service'],
    'TechSupport': ['No internet service'],
    'StreamingTV': ['No internet service'],
    'StreamingMovies': ['No internet service'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'tenure': [1],
    'MonthlyCharges': [29.85],
    'TotalCharges': [29.85]
})

# Ajustar el codificador
preprocessor.fit(dummy_data)

# Título de la aplicación
st.title("Churn Prediction App")

# Función para realizar predicciones
def predict_churn(data):
    try:
        # Transformar las variables categóricas
        data_encoded = preprocessor.transform(data)
        # Hacer la predicción
        prediction = loaded_model.predict(data_encoded)
        return prediction
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        return None

# Crear la interfaz de usuario con Streamlit
st.write("Introduce las características del cliente:")

# Entradas de las características
gender = st.selectbox('Gender', ['Male', 'Female'])
SeniorCitizen = st.selectbox('Senior Citizen', [0, 1])
Partner = st.selectbox('Partner', ['Yes', 'No'])
Dependents = st.selectbox('Dependents', ['Yes', 'No'])
PhoneService = st.selectbox('Phone Service', ['Yes', 'No'])
MultipleLines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
StreamingTV = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
StreamingMovies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
PaymentMethod = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
tenure = st.slider('Tenure (months)', 1, 72, 1)
MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, max_value=150.0, value=70.0)
TotalCharges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=70.0)

# Convertir las entradas en un formato adecuado para el modelo
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'tenure': [tenure],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

# Botón para predecir
if st.button("Predecir"):
    prediction = predict_churn(input_data)
    if prediction is not None:
        st.write(f"La predicción del modelo es: {'Churn' if prediction[0] == 1 else 'No Churn'}")
