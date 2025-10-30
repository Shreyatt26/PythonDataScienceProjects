## Predict the species of an iris flower (setosa, versicolor, virginica) from four measurements:
# sepal_length, sepal_width, petal_length, petal_width


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

## for visualization
import seaborn as sns
import matplotlib.pyplot as plt

## training/validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report

## XGBoost
from xgboost import XGBClassifier

## Load data
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target # 0=setosa, 1=versicolor, 2=virginica

#print(iris)

df = X.copy()
df["species"] = y
#print(df.head())
#df["species"].value_counts()

## Plots
#sns.pairplot(df, hue="species", vars=X.columns)
#plt.show()

## split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

## Baseline XGBoost model
xgb = XGBClassifier(objective = "multi:softprob", #returns class probabilities
                    num_class=3,
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1)

xgb.fit(X_train, y_train)
proba = xgb.predict_proba(X_val)
preds = proba.argmax(1)

print("Accuracy: ", accuracy_score(y_val, preds))
#print("LogLoss: ", log_loss(y_val, preds))
print(classification_report(y_val, preds, target_names=iris.target_names))