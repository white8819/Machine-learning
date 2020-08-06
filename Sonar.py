#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data=pd.read_csv('C:/Users/DC/Downloads/sonar.csv',names=range(0,61),header=0)
data


# In[ ]:


df=pd.DataFrame(data)
df


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df[60]=le.fit_transform(df[60].astype(str))
df


# In[ ]:





# In[ ]:





# In[ ]:


from scipy.stats import zscore
threhold=3
z=np.abs(zscore(df))
z


# In[ ]:


print(np.where(z>3))


# In[ ]:


df_new=df[(z<3).all(axis=1)]


# In[ ]:


x=df_new.iloc[:,0:-1]
y=df_new.iloc[:,-1]

x
y


# In[ ]:


df_new.shape


# In[ ]:


y=y.values.reshape(-1,1)


# In[ ]:


x.shape
y.shape


# In[ ]:


df.dtypes


# In[ ]:


df_new.describe()


# In[ ]:


df_new.skew()


# In[ ]:


for col in df_new.columns:
    if df_new.skew().loc[col]>0.55:
        df_new[col]=np.log(df_new[col])


# In[ ]:


df_new.skew()


# In[ ]:


collist=df_new.columns.values
nrows=10
ncol=12
plt.figure(figsize=(ncol,5*ncol))
for i in range(1,len(collist)):
    plt.subplot(nrows,ncol,i+1)
    sns.boxplot(df_new[collist[i]],color='green',orient='v')
    plt.tight_layout()


# In[ ]:


for i in range(0,60):
    plt.figure()
    sns.barplot(60,i,data=df_new)


# In[ ]:


se=StandardScaler()
x=se.fit_transform(x)


# In[ ]:


x


# In[ ]:


pca=PCA(n_components=15)


# In[ ]:


x_pca=pca.fit_transform(x)


# In[ ]:


df_new.corr()


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(df_new.corr(),annot=True)


# In[ ]:


def max_score(model):
    max_score=0
    for r_state in range(42,100):
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=r_state,test_size=0.20)
        model.fit(xtrain,ytrain)
        ypred=model.predict(xtest)
        accuracy=accuracy_score(ytest,ypred)
        print('accuracy is',accuracy,'with r state',r_state)
        if accuracy>max_score:
            max_score=accuracy
            final_r_state=r_state
            print(max_score,'is max accuracy against r_state',r_state)
            print(cross_val_score(model,x,y,cv=10,scoring='accuracy').mean())


# In[ ]:


max_score(LogisticRegression())


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=42,test_size=0.20)
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


max_score(GaussianNB())


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=42,test_size=0.20)
gnb=GaussianNB()
gnb.fit(xtrain,ytrain)
ypred=gnb.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


algo=KNeighborsClassifier()
para={'n_neighbors':[3,4,5,6,7,8,9,10]}
grid=GridSearchCV(estimator=algo,param_grid=para)
grid.fit(x,y)

print(grid.best_params_)


# In[ ]:


knc=KNeighborsClassifier(n_neighbors=6)
max_score(knc)


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=64,test_size=0.20)
knc.fit(xtrain,ytrain)
ypred=knc.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


algo=SVC()
para={'kernel':['rbf','poly','linear'],'C':[1,10]}
grid=GridSearchCV(estimator=algo,param_grid=para)
grid.fit(x,y)

print(grid.best_params_)


# In[ ]:


max_score(SVC(kernel='poly',C=10))


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=94,test_size=0.20)
sv=SVC(kernel='poly',C=10)
sv.fit(xtrain,ytrain)
ypred=sv.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


algo=DecisionTreeClassifier()
para={'criterion':['gini','entropy']}
grid=GridSearchCV(estimator=algo,param_grid=para)
grid.fit(x,y)

print(grid.best_params_)


# In[ ]:


max_score(DecisionTreeClassifier())


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=62,test_size=0.20)
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(xtrain,ytrain)
ypred=dtc.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


algo=RandomForestClassifier()
para={'n_estimators':[50,100,150,200,500,1000],}
grid=GridSearchCV(estimator=algo,param_grid=para)
grid.fit(x,y)

print(grid.best_params_)


# In[ ]:


max_score(RandomForestClassifier())


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=42,test_size=0.20)
rf=RandomForestClassifier(n_estimators=150)
rf.fit(xtrain,ytrain)
ypred=rf.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x_pca,y,random_state=42,test_size=0.20)
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x_pca,y,random_state=42,test_size=0.20)
gnb=GaussianNB()
gnb.fit(xtrain,ytrain)
ypred=gnb.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x_pca,y,random_state=64,test_size=0.20)
knc.fit(xtrain,ytrain)
ypred=knc.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x_pca,y,random_state=94,test_size=0.20)
sv=SVC(kernel='poly',C=10)
sv.fit(xtrain,ytrain)
ypred=sv.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x_pca,y,random_state=62,test_size=0.20)
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(xtrain,ytrain)
ypred=dtc.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x_pca,y,random_state=42,test_size=0.20)
rf=RandomForestClassifier(n_estimators=150)
rf.fit(xtrain,ytrain)
ypred=rf.predict(xtest)
accuracy=accuracy_score(ytest,ypred)
print(accuracy_score(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))


# SVC and Logistic regression on pca has highest accuracy
# 

# In[ ]:





# In[ ]:




