from pathlib import Path
import numpy as np, pandas as pd
rng=np.random.default_rng(42)
out=Path(__file__).resolve().parent/'data'; out.mkdir(exist_ok=True)
n=1600
neighborhood=rng.choice(['Central','North','South','East','West'],n,p=[.22,.18,.20,.18,.22])
quality=rng.integers(3,11,n)
living=rng.normal(1750,550,n).clip(600,4200).round().astype(int)
lot=rng.normal(8500,2800,n).clip(2500,22000).round().astype(int)
year=rng.integers(1950,2024,n)
rooms=np.clip((living/400+rng.normal(2.5,1,n)).round(),3,12).astype(int)
beds=np.clip((rooms*.55+rng.normal(0,1,n)).round(),1,6).astype(int)
baths=np.clip((quality/3+rng.normal(0,0.5,n)).round(),1,4).astype(int)
garage=np.clip((quality/2+rng.normal(0,0.7,n)).round(),0,3).astype(int)
basement=(living*rng.uniform(.25,.7,n)).round().astype(int)
renov=np.maximum(year,rng.integers(1970,2024,n))
base=35000+quality**2*4200+living*105+lot*2.4+rooms*5000+beds*3500+baths*12000+garage*14000+basement*22+(year-1950)*850+(renov-year)*500
neigh_bonus=pd.Series(neighborhood).map({'Central':42000,'North':18000,'South':5000,'East':12000,'West':26000}).to_numpy()
price=(base+neigh_bonus+rng.normal(0,30000,n)).clip(55000,850000).round(0)
df=pd.DataFrame({'Id':np.arange(1,n+1),'OverallQual':quality,'GrLivArea':living,'LotArea':lot,'YearBuilt':year,'YearRemodAdd':renov,'BedroomAbvGr':beds,'FullBath':baths,'GarageCars':garage,'TotalBsmtSF':basement,'TotRmsAbvGrd':rooms,'Neighborhood':neighborhood,'HouseStyle':rng.choice(['1Story','2Story','Split','Ranch'],n),'SaleCondition':rng.choice(['Normal','Partial','Abnormal'],n,p=[.82,.13,.05]),'SalePrice':price})
# Introduce realistic missing values and duplicates for cleaning practice.
for col in ['LotArea','YearRemodAdd','TotalBsmtSF','Neighborhood']:
    idx=rng.choice(df.index,size=int(.025*n),replace=False); df.loc[idx,col]=np.nan
df=pd.concat([df,df.iloc[[10,25,100]]],ignore_index=True)
df.to_csv(out/'train.csv',index=False)
print(out/'train.csv', df.shape)
