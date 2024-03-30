#!/usr/bin/env python
# coding: utf-8

# # Decison Tree

# In[36]:


#. loading the required libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn import tree
from sklearn.model_selection import train_test_split


# In[37]:


data = pd.read_csv('car_data.csv')


# In[40]:


data.head()


# In[38]:


data.info()


# In[13]:


data.head()

#. Purchase Decision (No = 0; Yes = 1)
#. Purchased is the target, trying to predict if a customer will buy a car based on their Gender, Age or Salary.


# In[41]:


data.isna().sum()


# In[42]:


data.describe().transpose()


# In[16]:


#. encoding Gender using one hot encoder

#
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#
#ohe = OneHotEncoder() # declares an instance of the object 
#ohe_data=ohe.fit_transform(data[['Gender']]).toarray() # applies the object to data 
#feature_labels = ohe.categories_ #. labels are stored here
#df_ohe = pd.DataFrame(ohe_data, columns = feature_labels)
#new_df_ohe = pd.concat([data, df_ohe], axis = 1) # combining the original dataframe (df) and df_ohe
#    
#new_df_ohe.drop('Gender', axis=1, inplace=True)
#
#new_df_ohe.head()


# In[43]:


data['Gender'].unique()


# In[46]:


le = LabelEncoder() #. declares an instance of the object

le_data = le.fit_transform(data[['Gender']]) #. applies object to data

df_le = pd.DataFrame(le_data, columns = ['GenderNew']) #. creates a dataframe
df_le.head()

new_df = pd.concat([data, df_le], axis = 1)


# In[44]:


df_le.head()


# In[47]:


new_df.head()


# In[48]:


new_df.drop('Gender', axis=1, inplace=True)


# In[49]:


new_df.head()


# In[50]:


sns.heatmap(new_df.corr(), annot = True)


# In[51]:


#. Building the model, splitting the dataset and scaling

X = new_df.drop('Purchased', axis = 1)
y = new_df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

X,y= X_train, y_train

#X_train, y_train = train_test_split(X,y, test_size=0.3, random_state=42)


# In[23]:


X_train


# In[24]:


X_test


# In[25]:


y_train


# In[26]:


y_test


# In[27]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)  
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns) 


# In[52]:


model = tree.DecisionTreeClassifier(criterion='gini') 
# for classification, here you can change the algorithm as gini or entropy (information gain) 
#by default it is gini
# model = tree.DecisionTreeRegressor() for regression

# Train the model using the training sets and check score
model.fit(X_train_scaled, y_train)

model.score(X_train_scaled, y_train)

#Predict Output
predicted = model.predict(X_test_scaled)


# In[29]:


#. Purchase Decision (No = 0; Yes = 1)

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(predicted, y_test)
names = np.unique(predicted)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
           xticklabels=names, yticklabels=names)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[30]:


from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
plot_roc_curve,roc_auc_score)

print(classification_report(y_test, predicted))


# In[31]:


print(accuracy_score(y_test, predicted))


# In[32]:


path = model.cost_complexity_pruning_path(X_train_scaled, y_train)

alphas = path['ccp_alphas']

alphas


# In[33]:


accuracy_train, accuracy_test = [],[]

for i in alphas:
    model = tree.DecisionTreeClassifier(ccp_alpha=i)
    
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    accuracy_train.append(accuracy_score(y_train,y_train_pred))
    accuracy_test.append(accuracy_score(y_test,y_test_pred))
    
    
plt.figure(figsize=(14,7))
sns.lineplot(y=accuracy_train,x=alphas,label="Train Accuracy")
sns.lineplot(y=accuracy_test,x=alphas,label="Test Accuracy")
plt.xticks(ticks=np.arange(0.00,0.25,0.01))
plt.show()


# In[34]:


model = tree.DecisionTreeClassifier(ccp_alpha=0.02,random_state=40)
model.fit(X_train_scaled, y_train)
y_train_pred= model.predict(X_train_scaled)
y_test_pred= model.predict(X_test_scaled)

print(accuracy_score(y_train,y_train_pred),accuracy_score(y_test,y_test_pred))


# In[35]:


#. Random Forest

from sklearn.ensemble import RandomForestClassifier

model1 = RandomForestClassifier()

model.fit(X_train_scaled,y_train) 
basem_preds = model.predict(X_test_scaled)     
print(classification_report(y_test,basem_preds)) 
ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test)
plt.show() 
plt.figure(figsize=(40,40), dpi = 200); 

