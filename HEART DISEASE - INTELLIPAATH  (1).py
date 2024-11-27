#!/usr/bin/env python
# coding: utf-8

# PROBLEM STATEMENT: You are the data scientist at a medical research facility. The facility wants you to
#  build a machine learning model to classify if the given data of a patient should tell
#  if the patient is at the risk of a heart attack.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv(r"C:\Users\DELL\Downloads\dataset.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


mean = df['target'].mean()


# In[6]:


mean


# In[7]:


max = df['target'].max()


# In[8]:


max


# In[9]:


min = df['target'].min()


# In[10]:


min


# In[11]:


quartile = pd.DataFrame(df, columns=['target'])


# In[12]:


quartile


# In[13]:


positive = df[df['target']==1]


# In[14]:


positive


# In[15]:


negative = df[df['target']==0]


# In[16]:


negative


# In[17]:


count = df['target'].value_counts()


# In[18]:


count.plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.xlabel("Heart Disease")
plt.ylabel("Counts")
plt.title("distribution of Heart Disease")


# The bar plot shows a slightly higher prevalence of individuals with heart disease 

# In[19]:


plt.figure(figsize=(10, 6))
colors = {'Yes': 'red', 'No': 'blue'}
plt.scatter(df['age'], df['target'], color = ['Red'])
plt.xlabel('Age')
plt.ylabel('Counts')
plt.title('Heart disease Vs Age')


# In[20]:


correlation_matrix = df.corr(numeric_only=True)


# In[21]:


sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, linewidths=.5).heatmap(df.corr(numeric_only=True, annot=True, cmap='coolwarm', fmt='.2f', square=True))


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay,precision_score, recall_score, f1_score


# In[23]:


X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
#independent variable
y= df['target']       #dependent variable


# In[24]:


#Divide the dataset into train and test sets (70:30 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[25]:


# Build the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[26]:


#Predict values on the test set
y_pred = model.predict(X_test)


# In[27]:


#Build the confusion matrix and get the accuracy score
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


# In[50]:


# Print accuracy score
print(f'accuracy: {accuracy:.2f}')
print("conf_matrix:")
print(conf_matrix)


# In[29]:


from sklearn.tree import DecisionTreeClassifier


# In[30]:


X = df.drop('target', axis=1)
y = df['target']


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[32]:


# Initialize the Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)


# In[33]:


# Fit the model on the training set
dt_model.fit(X_train, y_train)


# In[34]:


# Predict on the test set
y_pred = dt_model.predict(X_test)


# In[35]:


# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


# In[36]:


# Print results
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.2f}")


# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)


# In[39]:


# Fit the model on the training set
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)


# In[40]:


# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


# In[41]:


# Print results
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.2f}")


# The Random Forest model is the most beneficial for this project due to its highest accuracy (82%) and balanced confusion matrix. It outperforms both Logistic Regression and Decision Tree models, making it the best choice for accurately predicting heart disease outcomes in this dataset.

# In[42]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score


# In[43]:


# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[44]:


# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier()
}


# In[45]:


# Store results
results = {}

for name, model in classifiers.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    


# In[46]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
    
results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred
    }


# In[47]:


for name, metrics in results.items():
    print(f"Model: {name}")
    cm = confusion_matrix(y_test, metrics['y_pred'])
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, metrics['y_pred']))
    print("\n" + "="*50 + "\n")


# In[48]:


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

for name, metrics in results.items():
    cm = confusion_matrix(y_test, metrics['y_pred'])
    plot_confusion_matrix(cm, f'Confusion Matrix for {name}')


# In[ ]:




