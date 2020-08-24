import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing  import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sns.set()




data = pd.read_csv('/Users/kudumlu/Desktop/diabetes2.csv')
print(data.info())

print(data.describe())
print(data.isnull().sum())
print(data.head())

data = data.copy(deep = True)
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

data['Glucose'].fillna(data['Glucose'].mean(), inplace = True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace = True)
data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace = True)
data['Insulin'].fillna(data['Insulin'].median(), inplace = True)
data['BMI'].fillna(data['BMI'].median(), inplace = True)


X=data.iloc[:,:-1]
Y=data.iloc[:,[-1]]

print(X.head())
print(Y.head())

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size = 0.2 , random_state =0)
#614+154 = 768

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, Y_train.values.ravel())


Y_pred = knn.predict(X_test)
print(Y_pred)

print("Accuracy:" , metrics.accuracy_score(Y_test, Y_pred))

confusion_matrix(Y_test,Y_pred)
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
