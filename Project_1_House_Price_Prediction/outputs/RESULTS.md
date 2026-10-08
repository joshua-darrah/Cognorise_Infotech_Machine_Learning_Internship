# House Price Prediction — Results

Best model: **Linear Regression**

| Model             |     MAE |    RMSE |       R2 |
|:------------------|--------:|--------:|---------:|
| Linear Regression | 26869.5 | 33810.6 | 0.952921 |
| Random Forest     | 25889   | 33896.8 | 0.952681 |

## Key insights
- MAE and RMSE quantify average prediction error; lower is better.
- R² measures the proportion of price variance explained by the model; higher is better.
- The feature-importance output identifies the strongest predictors available to the fitted tree model.

## Business recommendations
1. Use the strongest model as decision support for preliminary property valuation.
2. Review influential property characteristics before pricing or investment decisions.
3. Monitor errors across neighborhoods and price ranges before deployment.
4. Re-train periodically as market conditions change.
