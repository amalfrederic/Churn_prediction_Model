# Customer Churn Prediction using Machine Learning

## Overview

This project predicts whether a telecom customer is likely to churn (leave the service) based on customer demographics, subscription details, billing information, and service usage patterns.

The objective is to help telecom companies identify customers at risk of churning and take proactive retention measures.

---

## Problem Statement

Customer churn is a major challenge for telecom companies. Acquiring new customers is often more expensive than retaining existing ones. By analyzing customer behavior and service usage data, machine learning models can be used to predict customers who are likely to discontinue their services.

This project builds a binary classification model that predicts:

- **0 → Customer will stay**
- **1 → Customer is likely to churn**

---

## Dataset

Dataset: Telco Customer Churn Dataset

The dataset contains information about 7,043 telecom customers and includes features such as:

- Gender
- Senior Citizen Status
- Partner Status
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Tech Support
- Streaming Services
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Churn Status

---

## Project Workflow

### 1. Data Collection

The Telco Customer Churn dataset was imported into a Pandas DataFrame for analysis and preprocessing.

### 2. Exploratory Data Analysis (EDA)

Performed detailed exploratory analysis to understand customer behavior and churn patterns.

Key analyses included:

- Churn distribution
- Contract type vs churn
- Monthly charges vs churn
- Tenure vs churn
- Payment method analysis
- Internet service analysis

### 3. Data Cleaning

The following preprocessing steps were performed:

- Removed `customerID`
- Converted `TotalCharges` from string to numeric
- Handled blank values in `TotalCharges`
- Removed rows containing missing values
- Converted target variable (`Churn`) into binary format
- Removed redundant categorical representations

### 4. Feature Engineering

Categorical variables were encoded using one-hot encoding.

Examples:

- Gender
- Contract Type
- Internet Service
- Payment Method
- Service-related features

### 5. Data Scaling

Numerical features were standardized using `StandardScaler`.

Features scaled:

- Tenure
- Monthly Charges
- Total Charges

### 6. Model Training

A Logistic Regression model was trained using:

- 80% Training Data
- 20% Testing Data

Stratified train-test split was used to preserve churn distribution.

---

## Model Performance

### Logistic Regression Results

| Metric | Value |
|----------|----------|
| Accuracy | 80.38% |
| Precision (Churn) | 65% |
| Recall (Churn) | 57% |
| F1 Score (Churn) | 61% |

### Confusion Matrix

| Actual / Predicted | No Churn | Churn |
|-------------------|-----------|--------|
| No Churn | 916 | 117 |
| Churn | 159 | 215 |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

## Project Structure

```text
Customer_Churn_Prediction/

├── data/
│   └── churn.csv
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── model.pkl
├── scaler.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## Running the Project

### Clone Repository

```bash
git clone <repository-url>
cd Customer_Churn_Prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

macOS/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## Future Improvements

- Hyperparameter tuning
- Random Forest and XGBoost comparison
- Feature importance analysis
- ROC-AUC evaluation
- Cloud deployment
- Real-time customer risk dashboard

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow, including data cleaning, exploratory data analysis, feature engineering, model training, evaluation, and deployment. The Logistic Regression model achieved over 80% accuracy and successfully identified customer churn patterns that can help businesses improve customer retention strategies.