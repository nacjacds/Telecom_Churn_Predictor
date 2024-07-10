# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Probar con un usuario
with open('models/gbc_pipeline_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
y_pred = loaded_model.predict(X_test[0:1])