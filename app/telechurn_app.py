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

# Definir las columnas categóricas y numéricas
categorical_features = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines',
                        'Dependents', 'Partner']
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Crear el transformador de columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough'  # Deja las columnas numéricas sin cambios
)

# Crear un DataFrame de ejemplo para ajustar el codificador
dummy_data = pd.DataFrame({
    'Contract': ['Month-to-month'],
    'InternetService': ['DSL'],
    'PaymentMethod': ['Electronic check'],
    'OnlineSecurity': ['Yes'],
    'OnlineBackup': ['Yes'],
    'DeviceProtection': ['Yes'],
    'TechSupport': ['Yes'],
    'StreamingTV': ['Yes'],
    'StreamingMovies': ['Yes'],
    'MultipleLines': ['No phone service'],
    'Dependents': ['No'],
    'Partner': ['Yes'],
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
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
partner = st.selectbox('Partner', ['Yes', 'No'])
tenure = st.slider('Tenure (months)', 0, 72, 1)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=150.0, value=70.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=70.0)

# Convertir las entradas en un formato adecuado para el modelo
input_data = pd.DataFrame({
    'Contract': [contract],
    'InternetService': [internet_service],
    'PaymentMethod': [payment_method],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'MultipleLines': [multiple_lines],
    'Dependents': [dependents],
    'Partner': [partner],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Botón para predecir
if st.button("Predecir"):
    prediction = predict_churn(input_data)
    if prediction is not None:
        st.write(f"La predicción del modelo es: {'Churn' if prediction[0] == 1 else 'No Churn'}")
