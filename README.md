# CreditWise Loan Approval Prediction

A machine learning project that predicts loan approval decisions using applicants' financial, demographic, employment, and credit-related information.

## 📌 Project Overview

Loan approval decisions depend on multiple applicant attributes such as income, credit score, existing loans, debt-to-income ratio, savings, collateral value, and demographic characteristics.

This project develops a machine learning pipeline to analyze these factors and predict whether a loan application is likely to be approved.

## 🎯 Objectives

- Perform exploratory data analysis on loan applicant data.
- Handle missing values and preprocess categorical and numerical features.
- Analyze relationships between applicant characteristics and loan approval.
- Train and evaluate machine learning classification models.
- Identify important factors influencing loan approval predictions.

## 📊 Dataset

The dataset contains **1,000 applicant records** and **20 features**, including:

- Applicant Income
- Coapplicant Income
- Employment Status
- Age
- Marital Status
- Dependents
- Credit Score
- Existing Loans
- DTI Ratio
- Savings
- Collateral Value
- Loan Amount
- Loan Term
- Loan Purpose
- Property Area
- Education Level
- Gender
- Employer Category
- Loan Approved

### Target Variable

`Loan_Approved`

The target represents whether the applicant's loan was approved.

## 🔍 Project Workflow

1. Data loading
2. Data quality and missing-value analysis
3. Exploratory Data Analysis (EDA)
4. Feature preprocessing
5. Encoding categorical variables
6. Train-test split
7. Model training
8. Model evaluation
9. Performance comparison
10. Prediction analysis

## 🤖 Machine Learning Models

The project evaluates classification models including:

- Logistic Regression
- [Model 2]
- [Model 3]

> Add only the models that you actually used in the notebook.

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | XX% | XX% | XX% | XX% |
| Model 2 | XX% | XX% | XX% | XX% |
| Model 3 | XX% | XX% | XX% | XX% |

> Replace the values above with the actual results from your notebook.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 📁 Project Structure

```text
CreditWise_Loan_Approval_Prediction/
│
├── loan_system.ipynb
├── loan_approval_data.csv
├── README.md
└── .gitattributes
