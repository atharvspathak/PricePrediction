import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

housing=pd.read_csv('data.csv')

train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]        #LOC STORE LINE OF GIVEN INDEX IN SET
    strat_test_set=housing.loc[test_index]

housing=strat_train_set.copy()

housing=strat_train_set.drop("MEDV",axis=1)#last cploumn

housing_labels=strat_train_set["MEDV"].copy()

corr_matrix=housing.corr()

imputer=SimpleImputer(strategy="median")

imputer.fit(housing)

X=imputer.transform(housing)

housing_tr=pd.DataFrame(X,columns=housing.columns)

my_pipline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler())
])

housing_num_tr=my_pipline.fit_transform(housing)
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)
#iterfarace

some_data=housing.iloc[:5]
preapre_data=my_pipline.transform(some_data)
print(list(preapre_data))



file=open('model.pkl','wb')
pickle.dump(model,file)

file.close()
