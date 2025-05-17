#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Import the necessary libraries

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline


# In[2]:


pip install ucimlrepo


# In[2]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# metadata 
print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables) 


# In[3]:


##Let's take a look at our data
##I see this is a dataset containing health information of various patients, "num" is the target

data = pd.concat([X, y], axis=1)

data.head()


# In[4]:


##All numerical, but from domain knowledge I am aware some of these such as sex is supposed to be categorical

data.info()


# In[5]:


##Very few missing data points, dropping them won't cause issues

null = data.isnull().sum()
duplicates = data.duplicated().sum()

print(f'Missing data: {null}')
print(f'Duplicates: {duplicates}')


# In[6]:


data = data.dropna()

print(data.isnull().sum())


# In[7]:


##Now let's take a look at the data satistically
##Again, based off research and domain knowledge, I notice a few unusual maxes in "chol" and "trestbps". I also
##see an imbalance within our target variable "num", switching this to binary is likely going to allow for enhanced
##model performance. I also have noticed in a lot of cases that a high standard deviation indicates the presence
##of outliers, so let's visualize to take a better look.


data.describe()


# In[8]:


##Visualizing the data
##We see some features without Normality due to outliers as we suspected

for col in data.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(data[col], bins=40, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[9]:


##Now for feature engineering

##Looking at the distribution of the graphs, I will eyeball the cut-offs of "chol" and "trestbps" 
##to get Normal distribution
data = data[data['chol'] <= 400]
data = data[data['trestbps'] <= 170]

##High cholestral is typically 240 and above, creating a feature for it might help indicate heart disease.
##Similar process for "trestbps" (resting heart rate), "thalach", and "oldpeak". Also, we binarize the target variable.
data['high_chol'] = (data['chol'] > 240).astype(int)
data['high_bp'] = (data['trestbps'] > 140).astype(int)
data['low_thalach'] = (data['thalach'] < 100).astype(int)
data['high_oldpeak'] = (data['oldpeak'] > 2.0).astype(int)
data['target'] = (data['num'] > 0).astype(int)

##We one-hot encode, now because this data is not ordinal, leaving it as label encoded is not optimal for the
##models I am going to use (RandomForest, XGBoost, SVM)
categorical_cols = ['cp', 'restecg', 'slope', 'thal', 'ca']
for col in categorical_cols:
    data[col] = data[col].astype('category')

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

##Looks a lot better satistically
data.describe()


# In[11]:


data.head()


# In[12]:


##Define the target variable and features used, note that because RandomForest and XGBoost are not sensitive to
##scaling, we input the features as is. However, for SVM we will have to scale our data

##We will also be using GridSearchCV to optimize the hyperparameters

target = data['target']

features = data[['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'high_chol','high_bp','low_thalach', 'high_oldpeak',
                 'cp_2', 'cp_3', 'cp_4','restecg_1', 'restecg_2','slope_2', 
                 'slope_3','thal_6.0', 'thal_7.0','ca_1.0', 'ca_2.0', 'ca_3.0']]

# Train/test split on 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

param_grid = {'n_estimators': [100, 200],'max_depth': [None, 10, 20],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,scoring='f1',n_jobs=-1,verbose=1)

grid_search.fit(X_train, y_train)


# In[13]:


##Printing results

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[14]:


##Feature importances to see what can be removed in the future and highlight what helps in indicating heart disease
##using this dataset.

importances = pd.Series(best_rf.feature_importances_, index=features.columns)
importances.sort_values(ascending=False).plot(kind='barh', figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()


# In[15]:


##ROC Curve graph

y_probs = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[18]:


##Let's try XGBoost next, GridSearch was not working for some reason so I used AI to help me manually input
##the hyperparameters

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, 
                                                    random_state=42)

xgb = XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,
    scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1],use_label_encoder=False,eval_metric='logloss',
    random_state=42)
xgb.fit(X_train, y_train)


# In[19]:


y_pred = xgb.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[20]:


y_probs = xgb.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"XGBoost ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend(loc='lower right')
plt.show()


# In[21]:


##Now for SVM

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, 
                                                    random_state=42)


# In[22]:


svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42))
])

svm_pipeline.fit(X_train, y_train)


# In[23]:


y_pred = svm_pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[24]:


y_probs = svm_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"SVM ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc='lower right')
plt.show()

