# Customer Churn Predictor for a Telecom Provider

This project aims to predict customer churn for a telecommunications company using machine learning models. Customer churn prediction helps the company identify customers who are likely to cancel their service, enabling proactive retention strategies.

## Problem Statement

The objective is to predict the likelihood of customer churn, defined as customers who leave the company's services. High churn rates can significantly impact revenue and profitability, making churn prediction a critical task.

## Project Description

This project aims to predict churn (customer attrition) in a telecommunications company. Predicting churn enables the company to take proactive measures to retain its customers, thereby improving customer satisfaction and reducing the churn rate.

## Dataset

The dataset used comes from a Californian internet service provider. This dataset includes various customer features, such as demographic information, service usage data, and account details.

## Data Preprocessing

1. **Data Cleaning**: Missing values were removed and outliers were handled.
2. **Categorical Variable Encoding**: Categorical and binary variables were converted to numeric format using one-hot encoding techniques.
3. **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the classes and improve the model's ability to predict the minority class.

## Models Used

1. **Decision Tree**: A model based on decision rules.
2. **Random Forest**: An ensemble of multiple decision trees.
3. **Gradient Boosting**: A sequentially improving model.
4. **SVM (Support Vector Machine)**: Finds an optimal hyperplane for class separation.
5. **ANN (Artificial Neural Network)**: A model inspired by the structure of the human brain.
6. **Logistic Regression**: A binary classification model based on the logistic function.

## Hyperparameter Optimization

An exhaustive search for the best hyperparameters was performed using GridSearchCV. The parameters adjusted include:

- **SMOTE**:
  - `sampling_strategy`: Sampling strategies (0.5 to 1.0)
  - `k_neighbors`: Number of neighbors (3 to 13)
- **Logistic Regression**:
  - `C`: Regularization parameter (0.0001 to 10000)
  - `penalty`: Types of penalty (`l1`, `l2`, `elasticnet`)
  - `solver`: Optimization algorithms (`lbfgs`, `saga`, `liblinear`, `newton-cg`)

## Model Validation

To ensure the robustness and generalizability of the model, k-fold cross-validation was used to evaluate the model's generalization ability and control overfitting. This technique divides the data into several subsets, improving the evaluation of the model's performance.

## Results

The Logistic Regression model, after being balanced, normalized, and optimized, showed superior performance in terms of Recall, achieving a score of 0.83. This high Recall indicates an excellent ability of the model to identify customers at risk of churn.

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
