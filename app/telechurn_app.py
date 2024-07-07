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
import subprocess
import sys

# Función para actualizar pip e instalar dependencias
def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Llamar a la función de instalación
install_packages()

# Importar librerías necesarias después de la instalación
from sklearn.ensemble import GradientBoostingClassifier

# Cargar el modelo
try:
    with open('models/modelo_gradientboosting.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except FileNotFoundError:
    st.error("El archivo del modelo no se encuentra en la ruta especificada.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Título de la aplicación
st.title("Churn Prediction App")

# Función para realizar predicciones
def predict_churn(data):
    try:
        prediction = loaded_model.predict(data)
        return prediction
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        return None

# Crear la interfaz de usuario con Streamlit
st.write("Introduce las características del cliente:")

# Ejemplo de características que podrías usar (ajusta según tu modelo)
gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', [0, 1])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.slider('Tenure (months)', 0, 72, 1)
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=150.0, value=70.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=70.0)

# Convertir las entradas en un formato adecuado para el modelo
input_data = pd.DataFrame({
    'gender': [gender],
    'senior_citizen': [senior_citizen],
    'partner': [partner],
    'dependents': [dependents],
    'tenure': [tenure],
    'phone_service': [phone_service],
    'multiple_lines': [multiple_lines],
    'internet_service': [internet_service],
    'online_security': [online_security],
    'online_backup': [online_backup],
    'device_protection': [device_protection],
    'tech_support': [tech_support],
    'streaming_tv': [streaming_tv],
    'streaming_movies': [streaming_movies],
    'contract': [contract],
    'paperless_billing': [paperless_billing],
    'payment_method': [payment_method],
    'monthly_charges': [monthly_charges],
    'total_charges': [total_charges]
})

# Botón para predecir
if st.button("Predecir"):
    prediction = predict_churn(input_data)
    if prediction is not None:
        st.write(f"La predicción del modelo es: {'Churn' if prediction[0] == 1 else 'No Churn'}")
