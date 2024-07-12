import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,roc_auc_score
from imblearn.over_sampling import SMOTE
import seaborn as sns


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

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#Model Instantiated And Trained
gnb = GaussianNB()
gnb.fit(X_train,y_train)
#Model predicted
y_train_pred=gnb.predict(X_train) 
y_test_pred=gnb.predict(X_test)

print(f"Train Accuracy Score:{accuracy_score(y_train,y_train_pred)}")
print(f"Test Accuracy Score:{accuracy_score(y_test,y_test_pred)}")

cm=confusion_matrix(y_test,y_test_pred)
print("----Confusion Matrix----")
print(f"True Positives:{cm[0,0]}")
print(f"True Negatives:{cm[1,1]}")
print(f"False Positive:{cm[0,1]}")
print(f"False Negative:{cm[1,0]}")

cm_matrix=x=pd.DataFrame(data=cm,columns=['Actual Positive:1 ','Actual Negative:0'],index=['Predict Positive:1','Predict Negative:0'])
sns.heatmap(cm_matrix,annot=True,fmt='d',cmap='YlGnBu')
plt.show()

print(f"----Classification Report----\n{classification_report(y_test,y_test_pred)}")

y_pred_prob= gnb.predict_proba(X_test)
y_pred_prob_df= pd.DataFrame(data=y_pred_prob, columns=['Prob of Class "bad" ','Prob of Class "good"'])

y_pred0=gnb.predict_proba(X_test)[:,0]

plt.rcParams['font.size']=12
plt.hist(y_pred0,bins=10)
plt.title('Histogram of Predicted Probabilities of CLass "bad"')
plt.xlim(0,1)
plt.xlabel('Predict probabilities of "bad" class')
plt.ylabel('Frequency')
plt.show()

y_pred1=gnb.predict_proba(X_test)[:,1]

plt.rcParams['font.size']=12
plt.hist(y_pred1,bins=10)
plt.title('Histogram of Predicted Probabilities of CLass "good"')
plt.xlim(0,1)
plt.xlabel('Predict probabilities of "good" class')
plt.ylabel('Frequency')
plt.show()

fpr,tpr,thresholds=roc_curve(y_test,y_pred1,pos_label="good")
plt.figure(figsize=(6,4))
plt.plot(fpr,tpr,linewidth=2)
plt.plot([0,1],[0,1],'k--')
plt.rcParams['font.size']=12
plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Credit Score')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Postive Rate (Sensitivity)')
plt.show()

ROC_AUC=roc_auc_score(y_test,y_pred1)
print("ROC AUC :",(ROC_AUC))

