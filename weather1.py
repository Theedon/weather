# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:15:30 2021

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer
#from sklearn.impute import CategoricalImputer

dataset= pd.read_csv(r"weatherAUS.csv").values
weather= pd.read_csv(r"weatherAUS.csv")
weather= weather.drop(["Date", "RainTomorrow"], 1)


Z= dataset
imputer= SimpleImputer(missing_values= np.nan, strategy= "most_frequent")
imputer.fit(Z[:, 2:23])
Z[:, 2:23]= imputer.transform(Z[:, 2:23])
dataset= Z

X= dataset[:, 1:22]
y= dataset[:, 22]




"""
#perform under_sampling
from imblearn.under_sampling import RandomUnderSampler
rus= RandomUnderSampler(random_state=0)
X_resampled, y_resampled= rus.fit_resample(X, y)
(X,y)= (X_resampled, y_resampled)
"""

from sklearn.model_selection import StratifiedKFold
skf= StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
    X_train, X_test= X[train_index], X[test_index]
    y_train, y_test= y[train_index], y[test_index]
    
(X,y)= (X_train, y_train)
    
#perform under_sampling
from imblearn.over_sampling import RandomOverSampler
ros= RandomOverSampler(random_state=0)
X_resampled, y_resampled= ros.fit_resample(X, y)
(X, y)= (X_resampled, y_resampled)


X_df =pd.DataFrame(X, columns= weather.columns)
X= X_df

Z= X
Z["Location"]= pd.get_dummies(Z["Location"])
Z["WindGustDir"]= pd.get_dummies(Z["WindGustDir"])
Z["WindDir9am"]= pd.get_dummies(Z["WindDir9am"])
Z["WindDir3pm"]= pd.get_dummies(Z["WindDir3pm"])
Z["RainToday"]= pd.get_dummies(Z["RainToday"])
X=Z

scaler = StandardScaler()
scaler.fit(X)
X= scaler.transform(X)

X_train, y_train= (X,y)









#TESTING ON MY OWN PREDICTIONS
"""
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)


from sklearn.model_selection import StratifiedKFold
skf= StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
    X_train, X_test= X[train_index], X[test_index]
    y_train, y_test= y[train_index], y[test_index]
    
"""



X_df =pd.DataFrame(X_test, columns= weather.columns)
X_test= X_df

Z= X_test
Z["Location"]= pd.get_dummies(Z["Location"])
Z["WindGustDir"]= pd.get_dummies(Z["WindGustDir"])
Z["WindDir9am"]= pd.get_dummies(Z["WindDir9am"])
Z["WindDir3pm"]= pd.get_dummies(Z["WindDir3pm"])
Z["RainToday"]= pd.get_dummies(Z["RainToday"])
X_test= Z

scaler = StandardScaler()
scaler.fit(X_test)
X_test= scaler.transform(X_test)









from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
#end of libraries
from sklearn import metrics
from sklearn.ensemble import VotingClassifier

#random forest classifier
randomForestModel= RandomForestClassifier(n_estimators= 120)
randomForestModel.fit(X_train, y_train)



#XGBClassifier
xgbModel= XGBClassifier()
xgbModel.fit(X_train, y_train)



ensembleRandomXg = VotingClassifier(estimators= [("Random Forest", randomForestModel), ("XG boost", xgbModel)], voting= "hard").fit(X_train,y_train)
prediction= ensembleRandomXg.predict(X_test)
#checking the accuracy
metrics.accuracy_score(y_test, prediction)#0.81

"""

#LGBMClassifier
lgbmModel= LGBMClassifier()
lgbmModel.fit(X_train, y_train)
prediction= lgbmModel.predict(X_test)
#checking the accuracy
metrics.accuracy_score(y_test, prediction)#0.65


#CatBoostClassifier
catModel= CatBoostClassifier()
catModel.fit(X_train, y_train)
prediction= catModel.predict(X_test)
#checking the accuracy
metrics.accuracy_score(y_test, prediction)#0.67

ensembleLightCatXg = VotingClassifier(estimators= [("Light Boost", lgbmModel), ("Cat boost", catModel), ("XG boost", xgbModel)], voting= "hard").fit(X_train,y_train)
prediction= ensembleLightCatXg.predict(X_test)
#checking the accuracy
metrics.accuracy_score(y_test, prediction)#0.66

ensembleRandomCat = VotingClassifier(estimators= [("Random Forest", randomForestModel), ("Cat boost", catModel)], weights=(1,7), voting= "soft").fit(X_train,y_train)
prediction= ensembleRandomCat.predict(X_test)
#checking the accuracy
metrics.accuracy_score(y_test, prediction)#0.69

ensembleXgbCat = VotingClassifier(estimators= [("XG boost", xgbModel), ("Cat boost", catModel)], weights=(7,1), voting= "soft").fit(X_train,y_train)
prediction= ensembleXgbCat.predict(X_test)
#checking the accuracy
metrics.accuracy_score(y_test, prediction)#0.66

ensembleCatXgb = VotingClassifier(estimators= [("XG boost", xgbModel), ("Cat boost", catModel)], weights=(1,7), voting= "soft").fit(X_train,y_train)
prediction= ensembleCatXgb.predict(X_test)
#checking the accuracy
metrics.accuracy_score(y_test, prediction)#0.67


ensembleRandomCatXg = VotingClassifier(estimators= [("Random Forest", randomForestModel), ("Cat boost", catModel), ("XG boost", xgbModel)], voting= "hard").fit(X_train,y_train)
prediction= ensembleRandomCatXg.predict(X_test)
#checking the accuracy
metrics.accuracy_score(y_test, prediction)#0.70

ensembleRandomLightCatXg = VotingClassifier(estimators= [("Random Forest", randomForestModel), ("Light Boost", lgbmModel), ("Cat boost", catModel), ("XG boost", xgbModel)], voting= "hard").fit(X_train,y_train)
prediction= ensembleRandomLightCatXg.predict(X_test)
#checking the accuracy
metrics.accuracy_score(y_test, prediction)#0.71

"""
