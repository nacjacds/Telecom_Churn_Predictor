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

# Cargar el modelo
with open('models/modelo_gradientboosting.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Título de la aplicación
st.title("Churn Prediction App")

# Función para realizar predicciones
def predict_churn(data):
    # Asegúrate de que los datos de entrada estén en el mismo formato que el conjunto de entrenamiento
    prediction = loaded_model.predict(data)
    return prediction

# Crear la interfaz de usuario con Streamlit
st.write("Introduce las características del cliente:")

# Ejemplo de entrada de usuario
# Ajusta los inputs según las características de tu modelo
feature_1 = st.number_input('Feature 1')
feature_2 = st.number_input('Feature 2')
# Añade más inputs según sea necesario

# Convertir las entradas en un formato adecuado para el modelo
input_data = np.array([[feature_1, feature_2]])  # Ajusta según las características necesarias

# Botón para predecir
if st.button("Predecir"):
    prediction = predict_churn(input_data)
    st.write(f"La predicción del modelo es: {'Churn' if prediction[0] == 1 else 'No Churn'}")
