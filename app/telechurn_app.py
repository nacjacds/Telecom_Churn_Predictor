import streamlit as st
import pickle
import numpy as np

# Cargar el modelo
model = pickle.load(open('modelo.pkl', 'rb'))

# Título de la aplicación
st.title("Mi Aplicación de Predicción")

# Entrada de datos del usuario
input_data = st.text_input("Introduce los datos de entrada (separados por comas):")

if st.button("Predecir"):
    input_data = np.array([float(i) for i in input_data.split(',')]).reshape(1, -1)
    prediction = model.predict(input_data)
    st.write(f"La predicción es: {prediction}")