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
    with open('models/modelo_gradientboosting.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except FileNotFoundError:
    st.error("El archivo del modelo no se encuentra en la ruta especificada.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Definir las categorías manualmente para cada característica
categorias = [
    ['Male', 'Female'],  # gender
    [0, 1],  # SeniorCitizen
    ['Yes', 'No'],  # Partner
    ['Yes', 'No'],  # Dependents
    list(range(0, 73)),  # tenure
    ['Yes', 'No'],  # PhoneService
    ['Yes', 'No', 'No phone service'],  # MultipleLines
    ['DSL', 'Fiber optic', 'No'],  # InternetService
    ['Yes', 'No', 'No internet service'],  # OnlineSecurity
    ['Yes', 'No', 'No internet service'],  # OnlineBackup
    ['Yes', 'No', 'No internet service'],  # DeviceProtection
    ['Yes', 'No', 'No internet service'],  # TechSupport
    ['Yes', 'No', 'No internet service'],  # StreamingTV
    ['Yes', 'No', 'No internet service'],  # StreamingMovies
    ['Month-to-month', 'One year', 'Two year'],  # Contract
    ['Yes', 'No'],  # PaperlessBilling
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],  # PaymentMethod
]

# Ajustar el codificador con las categorías conocidas
encoder = OneHotEncoder(categories=categorias, sparse_output=False, handle_unknown='ignore')

# Crear un DataFrame de ejemplo con todas las combinaciones posibles de categorías
dummy_data = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'tenure': [1],
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
    'MonthlyCharges': [29.85],
    'TotalCharges': [29.85]
})

# Ajustar el codificador
try:
    encoder.fit(dummy_data)
except ValueError as e:
    st.error(f"Error al ajustar el codificador: {e}")
    st.stop()

# Título de la aplicación
st.title("Churn Prediction App")

# Función para realizar predicciones
def predict_churn(data):
    try:
        # Codificar las variables categóricas
        data_encoded = encoder.transform(data)
        # Convertir a DataFrame para asegurarnos de que las columnas coinciden
        encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out())
        prediction = loaded_model.predict(encoded_df)
        return prediction
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        return None

# Crear la interfaz de usuario con Streamlit
st.write("Introduce las características del cliente:")

# Ejemplo de características que podrías usar (ajusta según tu modelo)
customer_id = st.text_input('Customer ID')
gender = st.selectbox('Gender', ['Male', 'Female'])
SeniorCitizen = st.selectbox('Senior Citizen', [0, 1])
Partner = st.selectbox('Partner', ['Yes', 'No'])
Dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.slider('Tenure (months)', 0, 72, 1)
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
MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, max_value=150.0, value=70.0)
TotalCharges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=70.0)

# Convertir las entradas en un formato adecuado para el modelo
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
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
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

# Codificar las variables categóricas
input_data_encoded = encoder.transform(input_data)

# Convertir a DataFrame para asegurarnos de que las columnas coinciden
input_data_encoded_df = pd.DataFrame(input_data_encoded, columns=encoder.get_feature_names_out())

# Botón para predecir
if st.button("Predecir"):
    prediction = predict_churn(input_data_encoded_df)
    if prediction is not None:
        st.write(f"La predicción del modelo es: {'Churn' if prediction[0] == 1 else 'No Churn'}")
