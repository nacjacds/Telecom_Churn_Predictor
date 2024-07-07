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
categorical_features = ['gender', 'Partner']
numeric_features = ['SeniorCitizen', 'tenure']

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
    'tenure': [1]
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
tenure = st.slider('Tenure (months)', 1, 72, 1)

# Convertir las entradas en un formato adecuado para el modelo
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'tenure': [tenure]
})

# Botón para predecir
if st.button("Predecir"):
    prediction = predict_churn(input_data)
    if prediction is not None:
        st.write(f"La predicción del modelo es: {'Churn' if prediction[0] == 1 else 'No Churn'}")
