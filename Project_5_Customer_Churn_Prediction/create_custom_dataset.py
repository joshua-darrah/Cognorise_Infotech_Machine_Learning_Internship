from pathlib import Path
import numpy as np, pandas as pd
rng=np.random.default_rng(43)
out=Path(__file__).resolve().parent/'data'; out.mkdir(exist_ok=True)
n=3500
contract=rng.choice(['Month-to-month','One year','Two year'],n,p=[.55,.25,.20])
internet=rng.choice(['Fiber optic','DSL','No'],n,p=[.45,.35,.20])
tech=rng.choice(['Yes','No'],n,p=[.62,.38])
payment=rng.choice(['Electronic check','Bank transfer','Credit card','Mailed check'],n,p=[.38,.25,.22,.15])
tenure=rng.integers(0,73,n)
monthly=np.clip(rng.normal(75,25,n),20,150).round(2)
total=(tenure*monthly+rng.normal(0,120,n)).clip(0).round(2)
support=rng.poisson(1.2,n)
services=rng.integers(1,7,n)
senior=rng.choice([0,1],n,p=[.82,.18])
partner=rng.choice(['Yes','No'],n,p=[.48,.52])
paper=rng.choice(['Yes','No'],n,p=[.59,.41])
logit=-2.6 + (contract=='Month-to-month')*1.25 + (contract=='One year')*.25 + (internet=='Fiber optic')*.35 + (tech=='No')*.55 + (payment=='Electronic check')*.35 + (tenure<12)*.95 + (tenure<6)*.55 + (monthly>90)*.45 + support*.18 + senior*.28 - (services*.16) - (contract=='Two year')*.55
prob=1/(1+np.exp(-logit))
churn=np.where(rng.random(n)<prob,'Yes','No')
df=pd.DataFrame({'customerID':[f'CUST{i:05d}' for i in range(1,n+1)],'gender':rng.choice(['Male','Female'],n),'SeniorCitizen':senior,'Partner':partner,'Dependents':rng.choice(['Yes','No'],n,p=[.3,.7]),'tenure':tenure,'PhoneService':rng.choice(['Yes','No'],n,p=[.9,.1]),'MultipleLines':rng.choice(['Yes','No','No phone service'],n,p=[.45,.45,.1]),'InternetService':internet,'OnlineSecurity':rng.choice(['Yes','No','No internet service'],n,p=[.45,.4,.15]),'OnlineBackup':rng.choice(['Yes','No','No internet service'],n,p=[.45,.4,.15]),'DeviceProtection':rng.choice(['Yes','No','No internet service'],n,p=[.45,.4,.15]),'TechSupport':tech,'StreamingTV':rng.choice(['Yes','No','No internet service'],n,p=[.45,.4,.15]),'StreamingMovies':rng.choice(['Yes','No','No internet service'],n,p=[.45,.4,.15]),'Contract':contract,'PaperlessBilling':paper,'PaymentMethod':payment,'MonthlyCharges':monthly,'TotalCharges':total,'SupportTickets':support,'ActiveServices':services,'Churn':churn})
for col in ['TotalCharges','MonthlyCharges','Contract','TechSupport']:
    idx=rng.choice(df.index,size=int(.02*n),replace=False); df.loc[idx,col]=np.nan
df=pd.concat([df,df.iloc[[5,55,555]]],ignore_index=True)
df.to_csv(out/'WA_Fn-UseC_-Telco-Customer-Churn.csv',index=False)
print(out/'WA_Fn-UseC_-Telco-Customer-Churn.csv', df.shape)
