import pandas as pd
df=pd.read_csv("C:\\Users\ABHISHEKXD\Downloads\Flood-Prediction-Model-master\kerala.csv")
print(df.head(5))

print("================================")

print(df.info())
#info() helps us to figure out which columns have null values and which don't have null values ,memory usage , and also shows the data types.
print("================================")

print(df.describe())
#Describe () help to find the statsical summary of data .

print("================================")

print(df.corr)

print("================================")

#replacing Value of Yes and no in the flood columns with 1/0

df["FLOODS"].replace(["YES","NO"],[1,0],inplace=True)

print(df.head())
#Hence we can now see that columns are changed with Classification of 1= Yes and 2= No.

print("================================================")

#Seperating the data which we are gonna use for prediction.
x=df.iloc[:,1:14]
print(x.head())

print("================================================")
y=df.iloc[:,15]
print(y.head())

print("================================================")

import matplotlib.pyplot as plt   
import numpy as np
# sets the backend of matplotlib to the 'inline' backend. 
                   
c = df[['JUN','JUL','AUG','SEP']]
c.hist()
plt.show()
#How the rainfall index vary during rainy season  

print("================================================")

from sklearn import preprocessing
#  Here preprocessing means that we are cleaning and organizing  the data for,
#  making suitbale for building and training machine learning
minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
#minmaxscaler is used to scale data in a dataset to a specified range using each fature minimum and maximum values.
minmax.fit(x).transform(x)
# Scaling the data between 0 and 1.

print("================================================")


#dividing the dataset into training dataset and test dataset. 
from sklearn import model_selection,neighbors
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.head())

print("________________________________________________________________")

print(y_train.head())

print("________________________________________________________________")
print(x_test.head())

print("\n")

print(y_test.head())


print("================================")

# type casting = changing the type of the data.

y_train=y_train.astype('int')
y_test=y_test.astype('int')
print(y_train)

print("================================")

#Now we are using the prediction alogo method
knn=neighbors.KNeighborsClassifier()
# KNeighborsClassifier algo uses proxiity to maake classifcation or prediction about the grouping of an individual data point
knn.fit(x_train,y_train)

print("================================")

# Predicted chance of Flood.
print("Predicted Values for the Floods:")
y_predict=knn.predict(x_test)
print(y_predict)
print("================================")
print("Actual Values for the Floods:")
print(y_test)
print("================================")
print("List of the Predicted Values:")
print(y_predict)
print("================================")
# Scaling the dataset.
from sklearn.model_selection import cross_val_score,cross_val_predict
#cross_val_score = This implies rather then splitting data into two parts only , 
#one to train and another to test on, the dataset is divided into more slices or folds.
x_train_std= minmax.fit_transform(x_train)
x_test_std= minmax.fit_transform(x_test)
knn_acc=cross_val_score(knn,x_train_std,y_train,cv=3,scoring='accuracy',n_jobs=-1)
knn_proba=cross_val_predict(knn,x_train_std,y_train,cv=3,method='predict_proba')

print("================================")
#How accurate is our model?
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
print("\nAccuracy Score:%f"%(accuracy_score(y_test,y_predict)*100))
print("Recall Score:%f"%(recall_score(y_test,y_predict)*100))
print("ROC score:%f"%(roc_auc_score(y_test,y_predict)*100))
print(confusion_matrix(y_test,y_predict))




