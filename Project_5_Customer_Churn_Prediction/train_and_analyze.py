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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

BASE=Path(__file__).resolve().parent; DATA=BASE/'data'/'WA_Fn-UseC_-Telco-Customer-Churn.csv'; OUT=BASE/'outputs'; MODELS=BASE/'models'
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
if not DATA.exists(): raise FileNotFoundError(f'Missing dataset: {DATA}')

df=pd.read_csv(DATA)
print(f'Dataset shape: {df.shape}')
df=df.drop_duplicates().copy()
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['SeniorCitizenLabel']=df['SeniorCitizen'].map({0:'No',1:'Yes'})
df['TenureGroup']=pd.cut(df['tenure'],[-1,6,12,24,48,72],labels=['0-6','7-12','13-24','25-48','49-72'])
df['ChurnFlag']=df['Churn'].map({'No':0,'Yes':1})
df['InactivityRisk']=((df['tenure']<=6) & (df['MonthlyCharges']>=df['MonthlyCharges'].median())).astype(int)

# EDA and business segmentation
plt.figure(figsize=(6,4)); sns.countplot(data=df,x='Churn',color='C0'); plt.title('Customer Churn Distribution'); plt.tight_layout(); plt.savefig(OUT/'01_churn_distribution.png',dpi=150); plt.close()
plt.figure(figsize=(8,5)); sns.boxplot(data=df,x='Churn',y='MonthlyCharges',color='C0'); plt.title('Monthly Charges by Churn'); plt.tight_layout(); plt.savefig(OUT/'02_monthly_charges_by_churn.png',dpi=150); plt.close()
seg=df.groupby('TenureGroup',observed=True)['ChurnFlag'].mean().mul(100).reset_index(name='ChurnRate')
seg.to_csv(OUT/'churn_by_tenure_group.csv',index=False)
plt.figure(figsize=(8,5)); sns.barplot(data=seg,x='TenureGroup',y='ChurnRate',color='C0'); plt.title('Churn Rate by Tenure Group'); plt.ylabel('Churn Rate (%)'); plt.tight_layout(); plt.savefig(OUT/'03_churn_by_tenure.png',dpi=150); plt.close()

# Model-ready data
X=df.drop(columns=['Churn','ChurnFlag','customerID','TotalCharges'],errors='ignore'); y=df['ChurnFlag']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
num=X.select_dtypes(include=np.number).columns.tolist(); cat=X.select_dtypes(exclude=np.number).columns.tolist()
prep=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced'),'Random Forest':RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42,n_jobs=-1)}
rows=[]
for name,m in models.items():
    pipe=Pipeline([('prep',prep),('model',m)]); pipe.fit(X_train,y_train); pred=pipe.predict(X_test); proba=pipe.predict_proba(X_test)[:,1]
    rows.append([name,accuracy_score(y_test,pred),precision_score(y_test,pred),recall_score(y_test,pred),f1_score(y_test,pred),roc_auc_score(y_test,proba)])
    joblib.dump(pipe,MODELS/(name.lower().replace(' ','_')+'.joblib'))
results=pd.DataFrame(rows,columns=['Model','Accuracy','Precision','Recall','F1','ROC_AUC']).sort_values('ROC_AUC',ascending=False)
results.to_csv(OUT/'model_results.csv',index=False); print('\nModel results:'); print(results.to_string(index=False))
plt.figure(figsize=(9,5)); plot_df=results.melt(id_vars='Model',value_vars=['Precision','Recall','F1','ROC_AUC']); sns.barplot(data=plot_df,x='Model',y='value',hue='variable',palette='deep'); plt.ylim(0,1); plt.title('Churn Model Comparison'); plt.tight_layout(); plt.savefig(OUT/'04_model_comparison.png',dpi=150); plt.close()

best_name=results.iloc[0]['Model']; best=joblib.load(MODELS/(best_name.lower().replace(' ','_')+'.joblib')); proba=best.predict_proba(X_test)[:,1]; pred=(proba>=0.5).astype(int)
cm=confusion_matrix(y_test,pred); pd.DataFrame(cm,index=['Actual No Churn','Actual Churn'],columns=['Predicted No Churn','Predicted Churn']).to_csv(OUT/'confusion_matrix.csv')
plt.figure(figsize=(5,4)); sns.heatmap(cm,annot=True,fmt='d',cbar=False,cmap='Blues'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Best Model Confusion Matrix'); plt.tight_layout(); plt.savefig(OUT/'05_confusion_matrix.png',dpi=150); plt.close()
est=best.named_steps['model']; names=best.named_steps['prep'].get_feature_names_out(); scores=pd.Series(np.abs(est.coef_[0]) if hasattr(est,'coef_') else est.feature_importances_,index=names); top=scores.sort_values(ascending=False).head(20); top.to_csv(OUT/'feature_importance.csv',header=['importance'])
plt.figure(figsize=(9,7)); sns.barplot(x=top.values,y=top.index,color='C0'); plt.title('Top Churn Risk Indicators'); plt.tight_layout(); plt.savefig(OUT/'06_top_churn_indicators.png',dpi=150); plt.close()

# Customer risk export for Power BI and business use.
scored=X_test.copy(); scored['ActualChurn']=y_test.values; scored['ChurnProbability']=proba; scored['RiskSegment']=pd.cut(scored['ChurnProbability'],[-.01,.30,.60,1.01],labels=['Active','At Risk','High Risk'])
scored.to_csv(OUT/'powerbi_customer_risk.csv',index=False)

summary=f'''# Customer Churn Prediction — Results\n\nDataset: Telco Customer Churn\n\nBest model by ROC-AUC: **{best_name}**\n\n{results.to_markdown(index=False)}\n\n## Segmentation\nCustomers are assigned to **Active**, **At Risk**, or **High Risk** using model churn probability thresholds of 0–0.30, 0.30–0.60 and 0.60–1.00.\n\n## Power BI\n`outputs/powerbi_customer_risk.csv` is ready to import into Power BI for churn rate, risk-segment, tenure and probability visuals.\n\n## Recommendations\n1. Prioritize high-risk customers for retention outreach.\n2. Create early-tenure onboarding interventions.\n3. Review pricing and service combinations associated with elevated churn.\n4. Track churn risk and retention outcomes over time.\n5. Use model scores as decision support and monitor model performance after deployment.\n'''
(OUT/'RESULTS.md').write_text(summary)
print(f'Best model: {best_name}')
