# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2.Read the data frame using pandas.
3.Get the information regarding the null values present in the dataframe.
4.Split the data into training and testing sets.
5.Convert the text data into a numerical representation using CountVectorizer.
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7.Finally, evaluate the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: A.ARUVI
RegisterNumber: 212222230014. 
*/
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
*/
```

## Output:
### Result Output:
![image](https://github.com/Anandanaruvi/Implementation-of-SVM-For-Spam-Mail-
### data.head()
![image](https://github.com/Anandanaruvi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120443233/7e2cf851-9505-4313-b718-8946c3b023e0)

### data.info()
![image](https://github.com/Anandanaruvi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120443233/a56e0a2c-9e4e-4241-99f8-446396a076c6)
### data.isnull().sum()
![image](https://github.com/Anandanaruvi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120443233/08730c4c-bef9-4730-8a9f-fa4bd3695d62)
### Y_prediction Value

![image](https://github.com/Anandanaruvi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120443233/3562f3d8-25fd-4720-81ff-0a5e8dae0832)
### Accuracy Value
![image](https://github.com/Anandanaruvi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120443233/528106ed-e569-4283-ab96-cd5f6e5d87f2)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
