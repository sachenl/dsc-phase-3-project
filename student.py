#!/usr/bin/env python
# coding: utf-8

# ## Final Project 3 Submission
# 
# Please fill out:
# * Student name: Zhiqiang Sun
# * Student pace: self paced
# * Scheduled project review date/time: 
# * Instructor name: 
# * Blog post URL:
# 

# # Business understanding
# 
# SyriaTel Customer Churn (Links to an external site.)
# Build a classifier to predict whether a customer will ("soon") stop doing business with SyriaTel, a telecommunications company. Note that this is a binary classification problem.
# 
# Most naturally, your audience here would be the telecom business itself, interested in losing money on customers who don't stick around very long. Are there any predictable patterns here?
# 
# 
# 

# # Plan
# Since the SyriaTel Customer Churn is a binary classification problem problem, I will try to use several different algorithms to fit the data and select one of the best one. The algorithms I will try include Logistic Regression, k-Nearest Neighbors, Decision Trees, Support Vector Machine. 
# The target of the data we need to fit is the column 'churn'.
# The features of the data is the other columns in dataframe. 
# However, when I load the data file into dataframe, i found some of the columns are linear correlated with each other. I need to drop one of them. We need to polish the data first. 

# In[137]:


#import all the necessary library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings


# In[59]:


df = pd.read_csv('bigml.csv')
df.head()


# In[60]:


df.info()


# Looking at the dataframe, I need to steply polish some features and remove some of the columns: 
# 1. The pairs of features inclued (total night minutes and total night charges), (total day minutes and total night charges), (total night minutes and total night charges), (total intl charge and total intl minutes) are high correlated with each other. I need to remove one in each columns. 
# 2. All the phone numbers are unique and act as id. So it should not related to the target. I will remove this feature.
# 3. The object columns will be catalized. 
# 

# In[61]:


to_drop = ['state', 'phone number', 'total day minutes', 'total night minutes', 'total night minutes' , 'total intl minutes']
df_polished = df.drop(to_drop, axis = 1)
df_polished.head()


# In[63]:


# The object features need to be catlized
to_cat_1 = [ 'international plan', 'voice mail plan' ]
df_cat = pd.DataFrame()
for col in to_cat_1:
    df_cat = pd.concat([df_cat, pd.get_dummies(df_polished[col], prefix=col, drop_first=True)], axis = 1)
df_cat.head()


# In[65]:


# The 'customer service calls' contains only 10 unique values. It need to be catlized too. 

to_cat_1 = ['customer service calls' ]
df_cat_2 = pd.get_dummies(df_polished['customer service calls'], prefix = 'customer service calls')
df_cat_2.head()


# In[66]:


df_polished_2 = pd.concat([df_polished, df_cat, df_cat_2], axis = 1)


# In[123]:


to_drop_2 = ['international plan', 'voice mail plan' ,'customer service calls', 'international plan', 'voice mail plan' ]
df_polished_3 = df_polished_2.drop(to_drop_2, axis=1)
df_polished_3.columns


# In[119]:


to_plot= ['account length', 'area code', 'number vmail messages', 'total day calls', 'total day charge',
       'total eve minutes', 'total eve calls', 'total eve charge',
       'total night calls', 'total night charge', 'total intl calls',
       'total intl charge']
fig, axes = plt.subplots(figsize = (15,15))
fig.suptitle('boxplot for continues features')
for idx, col in enumerate(to_plot):
    
    plt.subplot(4,3,idx+1)
    
    df_polished_3.boxplot(col)


# In[124]:


#It looks like most of the frames contain outlier values which may impact our fitting and predicting to the final results. We will try to remove the ouliers.
to_modify = ['account length', 'area code', 'number vmail messages', 'total day calls', 'total day charge',
       'total eve minutes', 'total eve calls', 'total eve charge',
       'total night calls', 'total night charge', 'total intl calls',
       'total intl charge']
df_polished_4 = df_polished_3.copy()
for col in to_modify:
    Q1 = df_polished_3[col].quantile(0.25)
    Q3 = df_polished_3[col].quantile(0.75)
    IQR = Q3 - Q1
    df_polished_4 = df_polished_4[(df_polished_3[col] >= Q1 - 1.5*IQR) & (df_polished_3[col] <= Q3 + 1.5*IQR)]


# In[125]:


to_plot= ['account length', 'area code', 'number vmail messages', 'total day calls', 'total day charge',
       'total eve minutes', 'total eve calls', 'total eve charge',
       'total night calls', 'total night charge', 'total intl calls',
       'total intl charge']
fig, axes = plt.subplots(figsize = (15,15))
fig.suptitle('boxplot for continues features')
for idx, col in enumerate(to_plot):
    
    plt.subplot(4,3,idx+1)
    
    df_polished_4.boxplot(col)


# # Now the data was ready and we need to prepare and modeling the data with varies models.

# In[ ]:


### Requirements

#### 1. Perform a Train-Test Split

For a complete end-to-end ML process, we need to create a holdout set that we will use at the very end to evaluate our final model's performance.

#### 2. Build and Evaluate several Model
##### For each of the model, we need several steps
    1. Build and Evaluate a base model
    2. Build and Evaluate Additional Logistic Regression Models
    3. Choose and Evaluate a Final Model
#### 3. Compare all the models and find the best model

#### 4. Choose and Evaluate a Final Model

Preprocess the full training set and test set appropriately, then evaluate the final model with various classification metrics in addition to log loss.


# #### 1.  Prepare the Data for Modeling
# The target is Cover_Type. In the cell below, split df into X and y, then perform a train-test split with random_state=42 and stratify=y to create variables with the standard X_train, X_test, y_train, y_test names.

# In[132]:


y = df_polished_4['churn']*1   #extract target and convert from boolen to int type
X = df_polished_4.drop('churn', axis= 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# Since the X features are in different scales, we need to make them to same scale.
# Now instantiate a StandardScaler, fit it on X_train, and create new variables X_train_scaled and X_test_scaled containing values transformed with the scaler.

# In[135]:



scale = StandardScaler()
scale.fit(X_train)
X_train_scaled = scale.transform(X_train)
X_test_scaled = scale.transform(X_test)


# In[136]:


X_train_scaled


# In[ ]:





# In[ ]:




