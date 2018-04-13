from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np 
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
Y = iris.target
#plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
#plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
#plt.scatter(X[100:150,0],X[100:150,1],color='yellow',marker='x',label='Virginia')
#plt.xlabel('petal length')
#plt.ylabel('sepal length')
#plt.legend(loc='upper left')
#plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40,eta0=0.1,random_state=0)
ppn.fit(X_test_std,Y_train)

Y_pred = ppn.predict(X_test_std)
print('Misclassified samples:%d' % (Y_test != Y_pred).sum())

