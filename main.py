
import pandas as pd
import numpy as np
import matplotlib.pyplot
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  GaussianNB
df=pd.read_csv('credit_score.csv')
df.head()
# For Easy Use, The Missing Values '?'  Changed  With 'NaN'
df.replace('?', np.nan, inplace=True)
#Gained Some Information About Dataset
df.info()
#Checked Missing Values
df.isna().sum().sort_values()
#Created X and y Values
X=df.drop(columns=['class'],axis=1)
y=df['class']
X['credit_amount'] = pd.to_numeric(X['credit_amount'], errors='coerce')
X['age'] = pd.to_numeric(X['age'], errors='coerce')
#Missing Values Filled
categorical_feature_mask = X.dtypes == object
categorical_columns=X.columns[categorical_feature_mask]
non_categorical_columns=X.columns[~categorical_feature_mask]
categorical_imputer=SimpleImputer(strategy="most_frequent")
non_categorical_imputer= SimpleImputer(strategy='median')
for i in categorical_columns:
    categorical_imputer.fit(X[[i]])
    X[[i]]=categorical_imputer.transform(X[[i]])
for i in non_categorical_columns:
    X[i]=non_categorical_imputer.fit_transform(X[[i]])
#OneHot Encoding Imputed For Categorical Features
encoder=ce.OneHotEncoder(cols=categorical_columns)
X=encoder.fit_transform(X)
#StandardScaler Imputed
columns=X.columns
scaler=StandardScaler()
X[['credit_amount','age']]=scaler.fit_transform(X[['credit_amount','age']])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
