# Employee Attrition Prediction — Results

Dataset: IBM HR Analytics Employee Attrition & Performance

Best model by ROC-AUC: **Random Forest**

| Model               |   Accuracy |   Precision |   Recall |       F1 |   ROC_AUC |
|:--------------------|-----------:|------------:|---------:|---------:|----------:|
| Random Forest       |   0.846939 |    0.625    | 0.106383 | 0.181818 |  0.797915 |
| Logistic Regression |   0.755102 |    0.345679 | 0.595745 | 0.4375   |  0.791369 |
| Decision Tree       |   0.738095 |    0.272727 | 0.382979 | 0.318584 |  0.563442 |

## Dataset findings
- Attrition rate: **16.12%**.
- No duplicate rows were found.
- Feature engineering included tenure ratio, income per year at company, satisfaction index and overtime flag.

## Recommendations
1. Prioritize supportive interventions for employees with high model-predicted risk.
2. Review overtime-heavy teams and workload distribution.
3. Monitor satisfaction and work-life-balance indicators.
4. Pay particular attention to early-tenure employees.
5. Use predictions for retention support, not punitive decisions.
