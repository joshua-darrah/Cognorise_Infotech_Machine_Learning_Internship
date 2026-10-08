# Customer Churn Prediction — Results

Dataset: Telco Customer Churn

Best model by ROC-AUC: **Logistic Regression**

| Model               |   Accuracy |   Precision |   Recall |       F1 |   ROC_AUC |
|:--------------------|-----------:|------------:|---------:|---------:|----------:|
| Logistic Regression |   0.67     |    0.386139 | 0.722222 | 0.503226 |  0.750803 |
| Random Forest       |   0.778571 |    0.581395 | 0.154321 | 0.243902 |  0.71792  |

## Segmentation
Customers are assigned to **Active**, **At Risk**, or **High Risk** using model churn probability thresholds of 0–0.30, 0.30–0.60 and 0.60–1.00.

## Power BI
`outputs/powerbi_customer_risk.csv` is ready to import into Power BI for churn rate, risk-segment, tenure and probability visuals.

## Recommendations
1. Prioritize high-risk customers for retention outreach.
2. Create early-tenure onboarding interventions.
3. Review pricing and service combinations associated with elevated churn.
4. Track churn risk and retention outcomes over time.
5. Use model scores as decision support and monitor model performance after deployment.
