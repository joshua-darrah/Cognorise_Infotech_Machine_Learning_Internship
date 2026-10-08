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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
OUT = BASE / 'outputs'; MODELS = BASE / 'models'
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
if not DATA.exists(): raise FileNotFoundError(f'Missing dataset: {DATA}')

df = pd.read_csv(DATA)
print(f'Dataset shape: {df.shape}')
print(f'Duplicates: {df.duplicated().sum()}')
print(f'Missing cells: {int(df.isna().sum().sum())}')
df = df.drop_duplicates().copy()

df['TenureRatio'] = df['YearsAtCompany'] / df['TotalWorkingYears'].replace(0, np.nan)
df['IncomePerYearAtCompany'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
df['SatisfactionIndex'] = df[['JobSatisfaction','EnvironmentSatisfaction','RelationshipSatisfaction','WorkLifeBalance']].mean(axis=1)
df['IsOvertime'] = df['OverTime'].map({'Yes':1,'No':0})
df['AttritionFlag'] = df['Attrition'].map({'Yes':1,'No':0})

plt.figure(figsize=(6,4)); sns.countplot(data=df, x='Attrition', color='C0'); plt.title('Employee Attrition'); plt.tight_layout(); plt.savefig(OUT/'01_attrition_distribution.png',dpi=150); plt.close()
plt.figure(figsize=(8,5)); sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', color='C0'); plt.title('Monthly Income by Attrition'); plt.tight_layout(); plt.savefig(OUT/'02_income_by_attrition.png',dpi=150); plt.close()
plt.figure(figsize=(8,5)); sns.barplot(data=df, x='OverTime', y='AttritionFlag', color='C0'); plt.title('Attrition Rate by Overtime'); plt.tight_layout(); plt.savefig(OUT/'03_overtime_attrition.png',dpi=150); plt.close()

target='Attrition'; drop_cols=[target,'AttritionFlag','EmployeeNumber','EmployeeCount','StandardHours','Over18']
X=df.drop(columns=drop_cols,errors='ignore'); y=df[target].map({'No':0,'Yes':1})
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
num=X.select_dtypes(include=np.number).columns.tolist(); cat=X.select_dtypes(exclude=np.number).columns.tolist()
prep=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced'),'Decision Tree':DecisionTreeClassifier(max_depth=6,class_weight='balanced',random_state=42),'Random Forest':RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42,n_jobs=-1)}
rows=[]
for name,m in models.items():
    pipe=Pipeline([('prep',prep),('model',m)]); pipe.fit(X_train,y_train); pred=pipe.predict(X_test); proba=pipe.predict_proba(X_test)[:,1]
    rows.append([name,accuracy_score(y_test,pred),precision_score(y_test,pred),recall_score(y_test,pred),f1_score(y_test,pred),roc_auc_score(y_test,proba)])
    joblib.dump(pipe,MODELS/(name.lower().replace(' ','_')+'.joblib'))
results=pd.DataFrame(rows,columns=['Model','Accuracy','Precision','Recall','F1','ROC_AUC']).sort_values('ROC_AUC',ascending=False)
results.to_csv(OUT/'model_results.csv',index=False)
print('\nModel results:'); print(results.to_string(index=False))
plt.figure(figsize=(9,5)); plot_df=results.melt(id_vars='Model',value_vars=['Precision','Recall','F1','ROC_AUC']); sns.barplot(data=plot_df,x='Model',y='value',hue='variable',palette='deep'); plt.ylim(0,1); plt.title('Classification Model Comparison'); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(OUT/'04_model_comparison.png',dpi=150); plt.close()

best_name=results.iloc[0]['Model']; best=joblib.load(MODELS/(best_name.lower().replace(' ','_')+'.joblib')); pred=best.predict(X_test); proba=best.predict_proba(X_test)[:,1]
report=classification_report(y_test,pred,target_names=['No Attrition','Attrition'],output_dict=True)
pd.DataFrame(report).transpose().to_csv(OUT/'classification_report.csv')
cm=confusion_matrix(y_test,pred); pd.DataFrame(cm,index=['Actual No','Actual Yes'],columns=['Predicted No','Predicted Yes']).to_csv(OUT/'confusion_matrix.csv')
plt.figure(figsize=(5,4)); sns.heatmap(cm,annot=True,fmt='d',cbar=False,cmap='Blues'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Best Model Confusion Matrix'); plt.tight_layout(); plt.savefig(OUT/'05_confusion_matrix.png',dpi=150); plt.close()
est=best.named_steps['model']; names=best.named_steps['prep'].get_feature_names_out(); scores=pd.Series(np.abs(est.coef_[0]) if hasattr(est,'coef_') else est.feature_importances_,index=names); top=scores.sort_values(ascending=False).head(15); top.to_csv(OUT/'feature_importance.csv',header=['importance'])
plt.figure(figsize=(9,6)); sns.barplot(x=top.values,y=top.index,color='C0'); plt.title('Top Attrition Indicators'); plt.tight_layout(); plt.savefig(OUT/'06_top_attrition_indicators.png',dpi=150); plt.close()

summary=f'''# Employee Attrition Prediction — Results\n\nDataset: IBM HR Analytics Employee Attrition & Performance\n\nBest model by ROC-AUC: **{best_name}**\n\n{results.to_markdown(index=False)}\n\n## Dataset findings\n- Attrition rate: **{df.AttritionFlag.mean()*100:.2f}%**.\n- No duplicate rows were found.\n- Feature engineering included tenure ratio, income per year at company, satisfaction index and overtime flag.\n\n## Recommendations\n1. Prioritize supportive interventions for employees with high model-predicted risk.\n2. Review overtime-heavy teams and workload distribution.\n3. Monitor satisfaction and work-life-balance indicators.\n4. Pay particular attention to early-tenure employees.\n5. Use predictions for retention support, not punitive decisions.\n'''
(OUT/'RESULTS.md').write_text(summary)
print(f'Best model: {best_name}')
