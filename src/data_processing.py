# Data
df = pd.read_csv('data/df.csv')

def convert_dataframe(df):
    # Reemplazar valores en blanco o espacios vacíos con NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Convertir columnas binarias
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
    df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
    df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})
    df['MultipleLines'] = df['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 2})
    df['OnlineSecurity'] = df['OnlineSecurity'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['OnlineBackup'] = df['OnlineBackup'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['DeviceProtection'] = df['DeviceProtection'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['TechSupport'] = df['TechSupport'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['StreamingTV'] = df['StreamingTV'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['StreamingMovies'] = df['StreamingMovies'].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Convertir columnas categóricas con One-Hot Encoding
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])

    # Eliminar columnas no numéricas o innecesarias
    df = df.drop(columns=['customer_id'])

    # Asegurarse de que todas las columnas sean numéricas
    df = df.apply(pd.to_numeric, errors='coerce')

    # Eliminar filas con valores NaN
    df.dropna(inplace=True)

    return df

# Cargar el dataframe completo
df = pd.read_csv('data/df.csv') 

# Convertir los datos (limpiar y transformar)
df_final = convert_dataframe(df)


# Dividir en Train y Test csv
dftrain, dftest = train_test_split(df_final, test_size=0.2, random_state=42)

# Guardar los dataframes en archivos CSV
dftrain.to_csv('data/train/train.csv', index=False)
dftest.to_csv('data/test/test.csv', index=False)

# Separar variables predictoras y variable objetivo
X_train = dftrain[['Contract_Month-to-month', 'Contract_Two year', 'tenure', 'InternetService_Fiber optic', 'PaymentMethod_Electronic check', 'MonthlyCharges', 'TotalCharges', 'OnlineSecurity', 'TechSupport', 'MultipleLines']]# ACCURACY 0.79, PRECISION 0.83, RECALL: 0.91(GBC)
y_train = dftrain['Churn']
X_test = dftest[['Contract_Month-to-month', 'Contract_Two year', 'tenure', 'InternetService_Fiber optic', 'PaymentMethod_Electronic check', 'MonthlyCharges', 'TotalCharges', 'OnlineSecurity', 'TechSupport', 'MultipleLines']]# ACCURACY 0.79, PRECISION 0.83, RECALL: 0.91(GBC)
y_test = dftest['Churn']