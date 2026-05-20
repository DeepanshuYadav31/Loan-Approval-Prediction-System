# 💰 Loan-Approval-Prediction-System

An advanced Machine Learning & Ensemble Learning based Loan Approval Prediction System built using Python, Scikit-Learn, XGBoost, LightGBM, Pandas, and Streamlit.

This project predicts whether a loan application will be Approved or Rejected based on applicant financial details, credit score, loan amount, assets, and employment information.

The system uses a powerful Stacking Ensemble Model combining multiple machine learning algorithms to achieve high prediction accuracy and better generalization.

---

## 🚀 Project Highlights

✅ End-to-End Machine Learning Project
✅ Exploratory Data Analysis (EDA)
✅ Data Preprocessing & Feature Engineering
✅ Multiple ML Models Training
✅ Stacking Ensemble Learning
✅ Streamlit Web Application
✅ Heuristic / Probabilistic Scoring System
✅ Real-Time Prediction Interface
✅ Model Comparison & Evaluation

---

## 📌 Problem Statement

Loan approval is one of the most important decision-making tasks in the banking and finance industry.

Manual verification of loan applications is:

Time-consuming
Error-prone
Inconsistent

This project automates the process using Machine Learning techniques to help predict whether a customer is eligible for a loan based on their financial profile.

---

## 🛠️ Technologies Used

|  Technology	  | Purpose                   |
|---            | ---                       |
| Python	      | Programming Language      |
| Pandas	      | Data Manipulation         |
| NumPy	        | Numerical Computation     |
| Matplotlib	  | Data Visualization        |
| Seaborn	      | Statistical Visualization |
| Scikit-Learn	| Machine Learning          |
| XGBoost	      | Boosting Algorithm        |
| Streamlit	    | Web Application           |
| Joblib	      | Model Saving              |

---

## 📂 Project Structure
```
Loan-Approval-Prediction-System/
│
├── Loan_Approval_Prediction_System.ipynb
├── model.py
├── loan_prediction_pipeline.pkl
├── status_encoder.pkl
├── requirements.txt
├── README.md
└── dataset/
    └── Loan_Approval_Dataset.csv
```


---

## 📊 Dataset Features

The following features are used for prediction:

|  Feature	                      | Description            |
|---                              | ---                    |
| self_employed	                  | Employment Status      |
| income_annum	                  | Annual Income          |
| loan_amount	                    | Requested Loan Amount  |
| loan_term	                      | Loan Duration          |
| cibil_score	                    | Credit Score           |
| residential_assets_value-Learn	| Residential Assets     |
| commercial_assets_value	        | Commercial Assets      |
| luxury_assets_value	            | Luxury Assets          |
| bank_asset_value	              | Savings & Investments  |

---

## 📈 Exploratory Data Analysis (EDA)

The project includes detailed visualizations such as:

Count Plots
Heatmaps
KDE Plots
Pairplots
Histograms
Boxplots
Model Accuracy Comparison Charts

EDA helps understand:

Loan approval patterns
Credit score impact
Income distributions
Asset relationships
Feature correlations

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

Handling categorical features
Label Encoding
Feature Selection
Removing unnecessary columns
Train-Test Split
Feature Engineering
Data Transformation

---

## 🤖 Machine Learning Models Used

The following models were trained and evaluated:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine (SVM)
XGBoost Classifier
LightGBM Classifier

---

## 📊 Input Features

The application takes the following inputs:

- Self Employed Status
- Annual Income
- Loan Amount Requested
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value

---

## 🧠 Prediction Logic

The application calculates a prediction score using:

- CIBIL Score as the most important factor
- Income compared with loan amount
- Total asset value as collateral support
- Self-employment risk adjustment
- Loan term risk adjustment
- Log transformation for income, loan amount, and asset values

If the final score is greater than or equal to the defined threshold, the loan is predicted as **Approved**.  
Otherwise, it is predicted as **Rejected**.

---
