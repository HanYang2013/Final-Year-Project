print X_iris.shape,y_iris.shape
# *** Spyder Python Console History Log ***
print)X_iris[0],y_iris[0])
print(X_iris[0],y_iris[0])
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
X,y=X_iris[:, :2],y_iris
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
print X_train.shape,y_train.shape
print (X_train.shape,y_train.shape)
scaler=preprocessing.StandardScaler().fit(X_train).fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
scaler=preprocessing.StandardScaler().fit(X_train)
X_test=scaler.transform(X_test)
X_train=scaler.transform(X_train)
import matplotlib.pyplot as plt
colors=['red','greenyellow','blue']
for i in xrange(len(colors)):    xs=X_train[:,0][y_train==i]    ys=X_train[:,1][y_train==i]    plt.scatter(xs,ys,c=colors[i])    
for i in range(len(colors)):    xs=X_train[:,0][y_train==i]    ys=X_train[:,1][y_train==i]    plt.scatter(xs,ys,c=colors[i])    
plt.legend(iris.target_names)
for i in range(len(colors)):   xs=X_train[:,0][y_train==i]   ys=X_train[:,1][y_train==i]   plt.scatter(xs,ys,c=colors[i])   plt.legend(iris.target_names)   plt.xlabel('Sepal lengtg')   plt.ylabel('Sepal width')   
from sklearn.linear_modelsklearn._model import SGDClassifier
from sklearn.linear_model import SGDClassifier
clf=SGDClassifier()
clf.fit(X_train,y_train)
print (clf.coef_)
print (clf.intercept_)
x_min,x_max=X_train[:,0].min()- .5,X_train[:, 0].max+.5
x_min,x_max=X_train[:,0].min()- 0.5,X_train[:, 0].max+0.5
x_min,x_max=X_train[:, 0].min()- 0.5, X_train[:, 0].max()+0.5
y_min,y_max=X_train[: ,1].min() - .5,X_train[:, 1].max() +.5
xs=np.arange(x_min,x_max,0.5)
fig,axes=plt.subplots(1,3)
fig.set_size_inches(10,6)
for i in [0,1,2]:    axes[i].set_aspect('equal')    axes[i].set_title('Class'+str(i)+'versus the rest')    axes[i].set_x_label('Sepal length')    axes[i].set_y_label('Sepal width')    axes[i].set_xlim(x_min,x_max)    axes[i].set_ylim(y_min,y_max)    sca(axes[i])    plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)    ys=(-clf.intercept_[i]-Xs*clf.coef_[i,0])/clf.coef_[i,1]    plt.plot(xs,ys,hold=True)    
for i in [0,1,2]:    axes[i].set_aspect('equal')    axes[i].set_title('Class'+str(i)+'versus the rest')    axes[i].set_xabel('Sepal length')    axes[i].set_ylabel('Sepal width')    axes[i].set_xlim(x_min,x_max)    axes[i].set_ylim(y_min,y_max)    sca(axes[i])    plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)    ys=(-clf.intercept_[i]-Xs*clf.coef_[i,0])/clf.coef_[i,1]    plt.plot(xs,ys,hold=True)    
for i in [0,1,2]:    axes[i].set_aspect('equal')    axes[i].set_title('Class'+str(i)+'versus the rest')    axes[i].set_xlabel('Sepal length')    axes[i].set_ylabel('Sepal width')    axes[i].set_xlim(x_min,x_max)    axes[i].set_ylim(y_min,y_max)    sca(axes[i])    plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)    ys=(-clf.intercept_[i]-Xs*clf.coef_[i,0])/clf.coef_[i,1]    plt.plot(xs,ys,hold=True)    
for i in [0,1,2]:    axes[i].set_aspect('equal')    axes[i].set_title('Class'+str(i)+'versus the rest')    axes[i].set_xlabel('Sepal length')    axes[i].set_ylabel('Sepal width')    axes[i].set_xlim(x_min,x_max)    axes[i].set_ylim(y_min,y_max)    plt.sca(axes[i])    plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)    ys=(-clf.intercept_[i]-Xs*clf.coef_[i,0])/clf.coef_[i,1]    plt.plot(xs,ys,hold=True)    
fig,axes=plt.subplots(1,3);
for i in [0,1,2]:   axes[i].set_aspect('equal')   axes[i].set_title('Class'+str(i)+'versus the rest')   axes[i].set_xlabel('Sepal length')   axes[i].set_ylabel('Sepal width')   axes[i].set_xlim(x_min,x_max)   axes[i].set_ylim(y_min,y_max)   plt.sca(axes[i])   plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)   ys=(-clf.intercept_[i]-Xs*clf.coef_[i,0])/clf.coef_[i,1]   plt.plot(xs,ys,hold=True)   
fig,axes=plt.subplots(1,3); for i in [0,1,2]:   axes[i].set_aspect('equal')   axes[i].set_title('Class'+str(i)+'versus the rest')   axes[i].set_xlabel('Sepal length')   axes[i].set_ylabel('Sepal width')   axes[i].set_xlim(x_min,x_max)   axes[i].set_ylim(y_min,y_max)   plt.sca(axes[i])   plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)   ys=(-clf.intercept_[i]-Xs*clf.coef_[i,0])/clf.coef_[i,1]   plt.plot(xs,ys,hold=True)
fig,axes=plt.subplots(1,3);
fig.set_size_inches(10,6)
for i in [0,1,2]:   axes[i].set_aspect('equal')   axes[i].set_title('Class'+str(i)+'versus the rest')   axes[i].set_xlabel('Sepal length')   axes[i].set_ylabel('Sepal width')   axes[i].set_xlim(x_min,x_max)   axes[i].set_ylim(y_min,y_max)   plt.sca(axes[i])   plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)   ys=(-clf.intercept_[i]-Xs*clf.coef_[i,0])/clf.coef_[i,1]   plt.plot(xs,ys,hold=True)   
fig,axes=plt.subplots(1,3);hold
fig,axes=plt.subplots(1,3);
hold(true)
plt.hold(True)
fig,axes=plt.subplots(1,3);
fig.set_size_inches(10,6)
for i in [0,1,2]:   axes[i].set_aspect('equal')   axes[i].set_title('Class'+str(i)+'versus the rest')   axes[i].set_xlabel('Sepal length')   axes[i].set_ylabel('Sepal width')   axes[i].set_xlim(x_min,x_max)   axes[i].set_ylim(y_min,y_max)   plt.sca(axes[i])   plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)   ys=(-clf.intercept_[i]-Xs*clf.coef_[i,0])/clf.coef_[i,1]   plt.plot(xs,ys,hold=True)   
fig,axes=plt.subplots(1,3);fig.set_size_inches(10,6)for i in [0,1,2]:   axes[i].set_aspect('equal')   axes[i].set_title('Class'+str(i)+'versus the rest')   axes[i].set_xlabel('Sepal length')   axes[i].set_ylabel('Sepal width')   axes[i].set_xlim(x_min,x_max)   axes[i].set_ylim(y_min,y_max)   plt.sca(axes[i])   plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)   ys=(-clf.intercept_[i]-Xs*clf.coef_[i,0])/clf.coef_[i,1]   plt.plot(xs,ys,hold=True)   
fig,axes=plt.subplots(1,3);fig.set_size_inches(10,6)for i in [0,1,2]:   axes[i].set_aspect('equal')   axes[i].set_title('Class'+str(i)+'versus the rest')   axes[i].set_xlabel('Sepal length')   axes[i].set_ylabel('Sepal width')   axes[i].set_xlim(x_min,x_max)   axes[i].set_ylim(y_min,y_max)   plt.sca(axes[i])   plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.prism)   ys=(-clf.intercept_[i]-xs*clf.coef_[i,0])/clf.coef_[i,1]   plt.plot(xs,ys,hold=True)   
runfile('C:/Users/hanyang/.spyder2-py3/temp.py', wdir='C:/Users/hanyang/.spyder2-py3')

##---(Sun Sep 18 21:26:39 2016)---
runfile('C:/Users/hanyang/.spyder2-py3/Classification.py', wdir='C:/Users/hanyang/.spyder2-py3')

##---(Mon Sep 19 22:00:27 2016)---
runfile('C:/Users/hanyang/.spyder2-py3/SVM.py', wdir='C:/Users/hanyang/.spyder2-py3')

##---(Fri Sep 23 20:24:08 2016)---
runfile('C:/Users/hanyang/.spyder2-py3/Classification.py', wdir='C:/Users/hanyang/.spyder2-py3')

##---(Wed Sep 28 21:30:33 2016)---
runfile('C:/Users/hanyang/.spyder2-py3/train1.py', wdir='C:/Users/hanyang/.spyder2-py3')

##---(Wed Sep 28 22:35:13 2016)---
runfile('C:/Users/hanyang/.spyder2-py3/train1.py', wdir='C:/Users/hanyang/.spyder2-py3')
import scipy.io as sio
train110='E:/EEG/train_1/1_1_0.mat'
data1=sio.loadmat(train110)
data1
print(data1)
train='E:/EEG/train_1'
data=sio.loadmat(train)
runfile('C:/Users/hanyang/.spyder2-py3/train1.py', wdir='C:/Users/hanyang/.spyder2-py3')
oac_struct=data1['dataStruct']
oac_struct.shape
val=oac_struct[0,0]
val
val['data']
runfile('C:/Users/hanyang/.spyder2-py3/train1.py', wdir='C:/Users/hanyang/.spyder2-py3')
import pywt
runfile('C:/Users/hanyang/.spyder2-py3/train1.py', wdir='C:/Users/hanyang/.spyder2-py3')

##---(Wed Oct  5 11:05:37 2016)---
from sklearn.decomposition import FastICA, PCA

##---(Fri Oct  7 16:54:38 2016)---
import pywt as pw
print pywt.families
print pw.families
print (pw.families)
print (pywt.families)
print (pw.families)
print (pw.wavelist(‘coif’))
import pywt
pywt.families()
pywt.wavelist('coif')
import scipy.io as sio
from pywt import wavedec
train110='E:/EEG/train_1/1_1_0.mat'
data1=sio.loadmat(train110)
oac_struct=data1['dataStruct']
val=oac_struct[0,0]
EEG=val['data']
coeffs = wavedec(EEG[:,0], 'db1', level=5)
cA5, cD5, cD4 ,cD3 ,cD2 ,cD1 = coeffs
runfile('C:/Users/hanyang/.spyder2-py3/train1.py', wdir='C:/Users/hanyang/.spyder2-py3')
import scipy.io as sio
runfile('C:/Users/hanyang/.spyder2-py3/train1.py', wdir='C:/Users/hanyang/.spyder2-py3')