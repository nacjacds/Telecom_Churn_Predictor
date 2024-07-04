# Customer Churn Prediction for a Telecom Provider

This project aims to predict customer churn for a telecommunications company using machine learning models. Customer churn prediction helps the company identify customers who are likely to cancel their service, enabling proactive retention strategies.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Evaluation](#evaluation)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Problem Statement

The objective is to predict the likelihood of customer churn, defined as customers who leave the company's services. High churn rates can significantly impact revenue and profitability, making churn prediction a critical task.

## Dataset

The dataset used in this project contains information about customers, such as their demographic details, service usage, and contract information. The key features include:

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

## Feature Engineering

### Steps:
1. **Handling Missing Values**: Replace missing values or spaces with NaN and then impute or drop them as necessary.
2. **Encoding Categorical Variables**: Convert binary and categorical variables into numerical format using techniques such as label encoding and one-hot encoding.
3. **Normalization**: Normalize features to ensure they are on a similar scale.

## Model Selection

Various machine learning models were tested, including:
- Decision Tree Classifier
- Support Vector Classifier (SVC)
- Random Forest Classifier
- Gradient Boosting Classifier

### Best Model

The best-performing model was the Gradient Boosting Classifier. This model was chosen based on its ability to handle class imbalance and its performance metrics.

## Evaluation

Models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Confusion Matrix

Confusion matrices were used to visualize the performance of models, highlighting the importance of recall in this context due to the high cost of false negatives (customers who churn but are predicted to stay).

## Results

The final Gradient Boosting Classifier model achieved:
- **Accuracy**: 78%
- **Precision**: 
    - No Churn: 83%
    - Churn: 62%
- **Recall**: 
    - No Churn: 89%
    - Churn: 50%
- **F1-Score**: 
    - No Churn: 86%
    - Churn: 55%

## How to Run

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/telecom-churn-prediction.git
    cd telecom-churn-prediction
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook**:
    ```sh
    jupyter notebook Tele_Churn3.ipynb
    ```

4. **Train and Evaluate the Model**:
    Follow the instructions in the Jupyter Notebook to preprocess the data, train the model, and evaluate its performance.

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- jupyter

Install the dependencies using:
```sh
pip install -r requirements.txt
