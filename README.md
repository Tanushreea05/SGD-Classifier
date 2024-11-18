# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load Data: Load the Iris dataset using load_iris() and create a pandas DataFrame with the feature names and target variable.

2. Prepare Features and Target: Split the DataFrame into features (X) and target (y) by dropping the target column from X.

3. Split Data: Use train_test_split to divide the dataset into training and testing sets with a test size of 20%.

4. Train Model: Initialize and fit a Stochastic Gradient Descent (SGD) classifier on the training data.

5. Evaluate Model: Predict the target values for the test set, calculate accuracy, and print the confusion matrix to assess the model's performance.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Tanushree.A
RegisterNumber:212223100057  
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
X = df.drop('target', axis = 1)
Y = df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
sgd = SGDClassifier(max_iter=1000, tol=1e-3)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.3f}")
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)
```

## Output:

## aaccuracy score:
![image](https://github.com/user-attachments/assets/1bf37881-b4f9-4502-ba91-61355469e6bc)

## confusion matrix:
![image](https://github.com/user-attachments/assets/5760f154-73b2-4f92-8883-4437087e1177)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
