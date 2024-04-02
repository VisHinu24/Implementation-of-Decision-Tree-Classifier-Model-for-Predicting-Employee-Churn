# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages.

2.Read the data set.

3.Apply label encoder to the non-numerical column inoreder to convert into numerical values.

4.Determine training and test data set.

5.Apply decision tree Classifier and get the values of accuracy and data prediction
```

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: H.Vishinu
RegisterNumber:  212223220124
*/
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## data.head():
![output1](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144244396/c1a39e4f-5582-45a1-8088-075de72c6466)


## data.info():
![output2](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144244396/d3763aa5-c0ce-4781-817f-a43b08eb784f)

## isnull() and sum()
![output3](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144244396/31a905a5-b298-49db-8175-37ccb39324b8)

## data value counts()
![output5](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144244396/62957e01-7c65-4267-9b7a-62949b4ae0a5)

## data.head() for salary
![output4](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144244396/75acfdab-ac86-48d9-b7a8-ba897863c39e)

## x.head()
![output6](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144244396/4be4cbb5-c916-4575-b4f5-3a75db42c9aa)

## Accuracy value
![output7](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144244396/a55ef98d-354a-41d3-9014-816df3301e6e)

## Data prediction
![output8](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144244396/973f9d83-2ec2-408c-b4ea-9b1c7a024fad)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
