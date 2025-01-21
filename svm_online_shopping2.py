# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:37:37 2025

@author: avina
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Create the dataset
df_os = pd.read_excel('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Foundational ML Algorithms II/YbzwCGnqTfS09yVqVb5H_online_shoppers_intention.xlsx')
df_os.columns
df_os.dtypes
df_os.shape
df_os.head()
df_os.describe()

print("\nMissing Values:\n", df_os.isnull().sum())

# Preprocessing
visitor_type_mapping = {'Returning_Visitor': 1, 'New_Visitor': 2, 'Other': 0}
df_os['VisitorType'] = df_os['VisitorType'].map(visitor_type_mapping)

for col in ['Weekend', 'Revenue']: 
    if df_os[col].dtype == 'bool':
        df_os[col] = df_os[col].astype(int)
        

df_os['Month'] = LabelEncoder().fit_transform(df_os['Month'])

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_os.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()



# Split features and target
X = df_os.drop(columns=['Revenue'])
y = df_os['Revenue']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(np.isinf(X).sum(axis=0))
print(X.isnull().sum())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# SVM with hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 1],
    'kernel': ['rbf', 'linear']
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, scoring='f1', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Output results
print("Best Parameters:", grid_search.best_params_)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print(f"F1 Score: {f1:.2f}")
