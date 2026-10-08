# Dataset Setup — Cognorise Infotech ML Internship

Only two datasets remain to be downloaded. The Employee Attrition dataset is already included in this package.

## 1. House Prices
Use the Kaggle competition:
`https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques`

Place the downloaded `train.csv` here:
`Project_1_House_Price_Prediction/data/train.csv`

## 2. Employee Attrition
Already supplied:
`Project_3_Employee_Attrition_Prediction/data/WA_Fn-UseC_-HR-Employee-Attrition.csv`

## 3. Telco Customer Churn
Use:
`https://www.kaggle.com/datasets/blastchar/telco-customer-churn`

Place:
`WA_Fn-UseC_-Telco-Customer-Churn.csv`
here:
`Project_5_Customer_Churn_Prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## After downloading the two remaining datasets
Run:
```bash
python run_all.py
```
This executes all three project pipelines. The first project will train the regression models, the second will train the attrition classifiers, and the third will train churn classifiers and export a Power BI-ready customer-risk CSV.
