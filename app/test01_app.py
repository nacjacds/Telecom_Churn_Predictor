import streamlit as st
import pickle
import numpy as np

# Load your model (assuming you have a model.pkl file)
with open('models/modelo_gradientboosting.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title('Telecom Churn Predictor TEST')

# Add input widgets
# For example, let's assume your model takes three features as input
feature1 = st.number_input('Feature 1')
feature2 = st.number_input('Feature 2')
feature3 = st.number_input('Feature 3')

# Button to trigger prediction
if st.button('Predict'):
    # Prepare the input array for the model
    input_data = np.array([[feature1, feature2, feature3]])
    
    # Get the prediction from the model
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f'The predicted value is: {prediction[0]}')
