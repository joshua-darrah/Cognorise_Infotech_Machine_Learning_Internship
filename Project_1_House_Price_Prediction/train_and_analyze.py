from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data' / 'train.csv'
OUT = BASE / 'outputs'; MODELS = BASE / 'models'
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f'Missing dataset: {DATA}\nDownload House Prices — Advanced Regression Techniques and place train.csv in data/.')

df = pd.read_csv(DATA)
print(f'Dataset shape: {df.shape}')
print(f'Duplicates: {df.duplicated().sum()}')
print(f'Missing cells: {int(df.isna().sum().sum())}')

# Clean exact duplicates; remaining missing values are handled inside the ML pipeline.
df = df.drop_duplicates().copy()

# EDA
plt.figure(figsize=(9,5)); sns.histplot(df['SalePrice'], kde=True, color='C0'); plt.title('Sale Price Distribution'); plt.tight_layout(); plt.savefig(OUT/'01_sale_price_distribution.png', dpi=150); plt.close()
num_cols = df.select_dtypes(include=np.number).columns
corr = df[num_cols].corr(numeric_only=True)['SalePrice'].sort_values(ascending=False).head(12)
plt.figure(figsize=(10,6)); sns.barplot(x=corr.values, y=corr.index, color='C0'); plt.title('Top Numeric Correlations with SalePrice'); plt.tight_layout(); plt.savefig(OUT/'02_top_price_correlations.png', dpi=150); plt.close()

X = df.drop(columns=['SalePrice','Id'], errors='ignore')
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(exclude=np.number).columns.tolist()
preprocess = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
}
results = []
for name, model in models.items():
    pipe = Pipeline([('preprocess', preprocess), ('model', model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append([name, mean_absolute_error(y_test,pred), mean_squared_error(y_test,pred)**0.5, r2_score(y_test,pred)])
    joblib.dump(pipe, MODELS/(name.lower().replace(' ','_')+'.joblib'))
results_df = pd.DataFrame(results, columns=['Model','MAE','RMSE','R2']).sort_values('RMSE')
results_df.to_csv(OUT/'model_results.csv', index=False)
print('\nModel results:'); print(results_df.to_string(index=False))

plt.figure(figsize=(8,5)); sns.barplot(data=results_df, x='Model', y='RMSE', color='C0'); plt.title('Model Comparison — RMSE'); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(OUT/'03_model_rmse_comparison.png', dpi=150); plt.close()

best_name = results_df.iloc[0]['Model']
best = joblib.load(MODELS/(best_name.lower().replace(' ','_')+'.joblib'))
estimator = best.named_steps['model']; feature_names = best.named_steps['preprocess'].get_feature_names_out()
if hasattr(estimator, 'feature_importances_'):
    imp = pd.Series(estimator.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    imp.to_csv(OUT/'feature_importance.csv', header=['importance'])
    plt.figure(figsize=(9,6)); sns.barplot(x=imp.values, y=imp.index, color='C0'); plt.title('Most Influential Features'); plt.tight_layout(); plt.savefig(OUT/'04_feature_importance.png', dpi=150); plt.close()

summary = f'''# House Price Prediction — Results\n\nBest model: **{best_name}**\n\n{results_df.to_markdown(index=False)}\n\n## Key insights\n- MAE and RMSE quantify average prediction error; lower is better.\n- R² measures the proportion of price variance explained by the model; higher is better.\n- The feature-importance output identifies the strongest predictors available to the fitted tree model.\n\n## Business recommendations\n1. Use the strongest model as decision support for preliminary property valuation.\n2. Review influential property characteristics before pricing or investment decisions.\n3. Monitor errors across neighborhoods and price ranges before deployment.\n4. Re-train periodically as market conditions change.\n'''
(OUT/'RESULTS.md').write_text(summary)
print(f'Best model: {best_name}')
