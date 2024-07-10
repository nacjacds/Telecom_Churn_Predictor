# Definir el pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalizar los datos
    ('classifier', GradientBoostingClassifier(random_state=42))  # Clasificador
])

# Entrenar el modelo usando el pipeline
pipeline.fit(X_train, y_train)

# Guardar el pipeline completo en un archivo
with open('models/gbc_pipeline_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Cargar el pipeline completo desde el archivo
with open('models/gbc_pipeline_model.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

# Predecir en el conjunto de prueba
y_pred = loaded_pipeline.predict(X_test)