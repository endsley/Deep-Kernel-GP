#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from dknet import NNRegressor
from dknet.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,CovMat,Scale
from dknet.optimizers import Adam,SciPyMin,SDProp, Adam2
from dknet.utils import load_mnist

from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel
import sklearn.metrics


np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)



#	Used for Regression, but can try to match label for classification with rounding
def Deep_Kernel_Gaussian_Process(x_train, y_train, x_test,y_test):
	layers=[]
	layers.append(Dense(64,activation='tanh'))
	layers.append(Dense(64,activation='tanh'))
	layers.append(Dense(20))
	layers.append(CovMat(alpha=0.3,var=1.0,kernel='rbf'))
	opt=SciPyMin('l-bfgs-b')
	
	opt=Adam(1e-3)
	batch_size=500
	gp=NNRegressor(layers,opt=opt,batch_size=batch_size,maxiter=500,gp=True,verbose=True)
	gp.fit(x_train,y_train)
		
	#Can extract mapping z(x) and hyperparams for use in other learning algorithm
	alph=gp.layers[-1].s_alpha
	var=gp.layers[-1].var
	
	A_full=gp.fast_forward(x_train)	
	kernel=ConstantKernel(var)*RBF(np.ones(1))+WhiteKernel(alph)

	A_test=gp.fast_forward(x_test)
	gp1=GaussianProcessRegressor(kernel,optimizer=None)

	if A_full.shape[0] > 1000:
		data_index = np.arange(0,A_full.shape[0])
		np.random.shuffle(data_index)
		ind = data_index[0:1000]
		gp1.fit(A_full[ind,:],y_train[ind,:])
	else:
		#gp1.fit(A_full[500,:],y_train[500,:])
		gp1.fit(A_full,y_train)

	mu,stdt=gp1.predict(A_test,return_std=True)
	labels = np.rint(mu)
	return labels

if __name__ == "__main__":
	(x_train,y_train),(x_test,y_test)=load_mnist(shuffle=True)
	x_train=x_train.reshape(-1,28*28)
	x_test=x_test.reshape(-1,28*28)	
	y_test=np.argmax(y_test,1).reshape(-1,1)
	y_train=np.argmax(y_train,1).reshape(-1,1)
	#	You can use any data, here we use mnist. x_train each row is a sample	

	test_labels = Deep_Kernel_Gaussian_Process(x_train, y_train, x_test,y_test)
	test_acc = sklearn.metrics.accuracy_score(test_labels, y_test)
	
	print("GP Regression:")
	print("Test Accuracy : ",  test_acc ) 
	import pdb; pdb.set_trace()

