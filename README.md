# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Preprocess Data: Clean emails and convert text to numeric features (TF-IDF or BoW).
2. Split Dataset: Divide data into training and testing sets.
3. Train Model: Use SVM classifier (linear/RBF kernel) on training data.
4. Evaluate Model: Test performance using accuracy, precision, recall, and F1-score.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: MOHAMMED PARVEZ S
RegisterNumber: 212223040113
```

```PYTHON
import chardet
file="spam.csv"
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
```
<img width="571" height="29" alt="image" src="https://github.com/user-attachments/assets/c41d6053-afa8-4c42-a67d-a86ee57394f4" />

```PYTHON
import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')
data.head()
```
<img width="554" height="129" alt="image" src="https://github.com/user-attachments/assets/6315d377-efcf-42e3-a4a1-42bce9a5346c" />

```PYTHON
data.isnull().sum()
```
<img width="196" height="91" alt="image" src="https://github.com/user-attachments/assets/3e8ddfe2-724c-408e-80ef-adb5fec33617" />

```PYTHON
data.info()
```
<img width="271" height="171" alt="image" src="https://github.com/user-attachments/assets/09d067e2-79ec-4d64-a55c-92c7e420ffa8" />

```PYTHON
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train
```
<img width="878" height="107" alt="image" src="https://github.com/user-attachments/assets/e65d8766-4a0a-4f69-8c00-d03eff9058b6" />

```PYTHON
x_test
```
<img width="884" height="151" alt="image" src="https://github.com/user-attachments/assets/a7d1e9ac-98c2-4c14-803d-67635cb6bf23" />

```PYTHON
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train
```
<img width="502" height="31" alt="image" src="https://github.com/user-attachments/assets/27d3bc6a-04d1-4045-8f9f-c0463e73c50e" />

```PYTHON
x_test
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
cr=metrics.classification_report(y_test,y_pred)
print("Classification report:")
print(cr)
cm=metrics.confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
print(cm)
```

## Output:
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d2fc9f48-fc74-4d49-8ac5-6a9d3598765d" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
