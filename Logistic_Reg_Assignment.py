#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


train_df = pd.read_csv(r'C:\Users\Admin\Desktop\data science assighnments\Logistic Regression (1)\Logistic Regression\Titanic_train.csv')
test_df = pd.read_csv(r'C:\Users\Admin\Desktop\data science assighnments\Logistic Regression (1)\Logistic Regression\Titanic_test.csv')


# In[3]:


train_df.head()


# In[4]:


test_df.head()


# In[5]:


train_df.describe()


# In[6]:


test_df.describe()


# In[7]:


si = SimpleImputer(strategy='mean')


# In[8]:


train_df.iloc[:, 5] = si.fit_transform(train_df.iloc[:, [5]])


# In[9]:


train_df[['Age']] = si.fit_transform(train_df[['Age']])


# In[10]:


train_df.describe()


# In[11]:


test_df.iloc[:, 5] = si.fit_transform(test_df.iloc[:,[5]])


# In[12]:


test_df[['Age']] = si.fit_transform(test_df[['Age']])


# In[13]:


test_df.iloc[:, 8] = si.fit_transform(test_df.iloc[:,[8]])


# In[14]:


test_df[['Fare']] = si.fit_transform(test_df[['Fare']])


# In[15]:


test_df.describe()


# In[16]:


train_df.columns


# In[17]:


test_df.columns


# In[18]:


# Step 1: Identify irrelevant columns (these can be specific to your data)
irrelevant_columns = ['Pclass', 'Name','Ticket','Fare','Cabin','Embarked',]  # Modify this list

# Step 2: Remove irrelevant columns from both datasets
train_cleaned = train_df.drop(columns=irrelevant_columns, axis=1)
test_cleaned = test_df.drop(columns=irrelevant_columns, axis=1)


# In[19]:


train_cleaned


# In[20]:


# One-Hot Encoding using pandas `get_dummies`
train_cleaned = pd.get_dummies(train_cleaned, columns=['Sex'], drop_first=False)
train_cleaned


# In[21]:


from sklearn.model_selection import train_test_split ,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from scipy.stats import uniform, loguniform


# In[46]:


train_cleaned.columns
train_cleaned.head(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:


# Step 1: Select the features (X) and target variable (y)
X = train_cleaned[['PassengerId','Age', 'SibSp', 'Parch','Sex_female','Sex_male']]  # Use the features except 'Survived'
y = train_cleaned['Survived']  # The target variable


# In[48]:


# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[49]:


model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print(model)


# In[50]:


y_pred = model.predict(X_test_scaled)


# In[28]:


#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[29]:


print(f'Accuracy: {accuracy:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')


# In[30]:


# Regularization - Trying a different value of C (Regularization strength)
logreg = LogisticRegression(C=0.1)  # Smaller C means stronger regularization
logreg.fit(X_train_scaled, y_train)

# Predict and evaluate again
y_pred = logreg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy with C=0.1: {accuracy:.4f}')


# In[31]:


# Define the parameter space for RandomizedSearchCV
param_dist = {
    'C': loguniform(1e-4, 1e4),  # Log-uniform distribution for C (regularization strength)
    'solver': ['liblinear','lbfgs']  # Different solvers to try
}


# In[32]:


# Perform Randomized Search Cross Validation
random_search = RandomizedSearchCV(logreg, param_distributions=param_dist, n_iter=25, cv=3, verbose=1, random_state=42, n_jobs=1)

# Fit the model
random_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = random_search.best_params_
print(f"Best Hyperparameters: {best_params}")


# In[33]:


# Initialize the model with best hyperparameters
best_logreg = LogisticRegression(C=0.09915644566638389, solver='liblinear')


# In[34]:


# Train the model
best_logreg.fit(X_train_scaled, y_train)


# In[35]:



# Evaluate the model
y_pred = best_logreg.predict(X_test_scaled)


# In[36]:


print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[37]:


get_ipython().system('pip install imbalanced-learn')


# In[38]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# In[39]:


# 1. Random Oversampling (Increase the minority class)
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)


# In[40]:



# 2. Random Undersampling (Reduce the majority class)
rus = RandomUnderSampler(random_state=42)
X_train_resampled2, y_train_resampled2 = rus.fit_resample(X_train, y_train)


# In[41]:


# Logistic Regression with Oversampled Data
model_ros = LogisticRegression(max_iter=200)
model_ros.fit(X_train_resampled, y_train_resampled)
y_pred_ros = model_ros.predict(X_test)


# In[42]:


# Logistic Regression with Undersampled Data
model_rus = LogisticRegression(max_iter=200)
model_rus.fit(X_train_resampled2, y_train_resampled2)
y_pred_rus = model_rus.predict(X_test)


# In[43]:


# Evaluate with Random Oversampling
print("Random Oversampling Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ros)}")
print("Classification Report:\n", classification_report(y_test, y_pred_ros))


# In[44]:


# Evaluate with Random Undersampling
print("Random Undersampling Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rus)}")
print("Classification Report:\n", classification_report(y_test, y_pred_rus))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




