#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


# In[7]:


df = pd.read_csv('/kaggle/input/d/kenanavar/red-wine-quality/winequality-red.csv', sep=';')


# In[8]:


df.head()


# ## EDA

# In[9]:


df.describe()


# In[10]:


df.info()


# In[25]:


df.columns.tolist()


# In[19]:


df.nunique()


# In[11]:


df['quality'].unique()


# In[13]:


df['quality'].value_counts()


# ## Data Visualization

# In[17]:


sns.countplot(x=df['quality'])
plt.show()


# In[22]:


sns.histplot(x=df['pH'], hue=df['quality'],palette='viridis')
plt.show()


# In[24]:


sns.histplot(x=df['alcohol'],bins=20,kde=True)
plt.show()


# In[28]:


df_corr = abs(df.corr())
plt.figure(figsize=(12,12))
sns.heatmap(df_corr, annot=True, vmin=0, vmax=1, cmap='Reds', fmt='.2f')
plt.show()


# In[30]:


df_corr = df.corrwith(df['quality']).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(df_corr , columns=['correlation']), annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5)
plt.title('Correlation with Quality')
plt.show()


# In[32]:


X=df.drop('quality',axis=1)
y=df['quality']


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)


# #### First Model with Decision Tree

# In[36]:


DT_model = DecisionTreeClassifier()
DT_model.fit(X_train,y_train)


# In[37]:


DT_model.score(X_test,y_test)


# In[38]:


DT_pred = DT_model.predict(X_test)


# In[40]:


DT_acc = accuracy_score(y_test ,DT_pred)
DT_acc


# In[41]:


print(classification_report(y_test , DT_pred))


# In[42]:


cm2 = confusion_matrix(y_test,DT_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm2,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[43]:


DT_r2 = r2_score(y_test, DT_pred)
mse = mean_squared_error(y_test, DT_pred)
rmse = np.sqrt(mse)
print("R2 Score :", DT_r2, "\n RMSE :", rmse)


# In[44]:


DT_pred_rounded = np.round(DT_pred).astype(int)


# In[45]:


df_comp_DT = pd.DataFrame({'Real Outcome' : y_test, 'Predicted Outcome' : DT_pred_rounded})
print(df_comp_DT)


# In[47]:


RF_model = RandomForestClassifier()
RF_model.fit(X_train,y_train)


# In[49]:


RF_model.score(X_test,y_test)


# In[48]:


RF_pred = RF_model.predict(X_test)


# In[50]:


print(classification_report(y_test , RF_pred))


# In[51]:


RF_r2 = r2_score(y_test, RF_pred)
mse = mean_squared_error(y_test, RF_pred)
rmse = np.sqrt(mse)
print("R2 Score :", RF_r2, "\n RMSE :", rmse)

