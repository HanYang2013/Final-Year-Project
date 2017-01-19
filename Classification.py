# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets
iris=datasets.load_iris()
X_iris,y_iris=iris.data,iris.target
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
X,y=X_iris[:, :2],y_iris
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
scaler=preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
import matplotlib.pyplot as plt
colors=['red','greenyellow','blue']
for i in range(len(colors)):
   xs=X_train[:,0][y_train==i]
   ys=X_train[:,1][y_train==i]
   plt.scatter(xs,ys,c=colors[i])
   plt.legend(iris.target_names)
   plt.xlabel('Sepal lengtg')
   plt.ylabel('Sepal width')
from sklearn.linear_model import SGDClassifier
clf=SGDClassifier(shuffle=False)
clf.fit(X_train,y_train)
print (clf.coef_)
print (clf.intercept_)
x_min,x_max=X_train[:, 0].min()- 0.5, X_train[:, 0].max()+0.5
y_min,y_max=X_train[: ,1].min() - .5,X_train[:, 1].max() +.5
import numpy as np
xs=np.arange(x_min,x_max,0.5)
fig,axes=plt.subplots(1,3)
fig.set_size_inches(10,6)
for i in [0,1,2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class'+str(i)+'versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min,x_max)
    axes[i].set_ylim(y_min,y_max)
    plt.sca(axes[i])
    plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)
    ys=(-clf.intercept_[i]-xs*clf.coef_[i,0])/clf.coef_[i,1]
    plt.plot(xs,ys,hold=True)
print (clf.predict(scaler.transform([[4.7, 3.1]])))
from sklearn import metrics
y_train_pred=clf.predict(X_train)
print (metrics.accuracy_score(y_train,y_train_pred))
print (clf.decision_function(scaler.transform([4.7,3.1])))
from sklearn import metrics
y_train_pred=clf.predict(X_train)
print (metrics.accuracy_score(y_train,y_train_pred))
y_pred=clf.predict(X_test)
print (metrics.accuracy_score(y_test,y_pred))