#!/usr/bin/python3
#-*- coding: utf-8 -*-

import numpy as np
import sys

threshold = 1e-10
epsilon = 1e-10
np.set_printoptions(suppress=True, precision=3, threshold=np.inf)

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def GMF_RealValue(Y, K, n_iter=10000, iter_per_epoch=100, lr=.001):
	M, N = Y.shape

	P = np.random.normal(loc=0, scale=1.0/K, size=(M, K))
	Q = np.random.normal(loc=0, scale=1.0/K, size=(N, K))
#H = np.random.normal(loc=0, scale=1.0/K, size=K)
	H = np.diag(np.ones(K))
	dP = np.zeros_like(P)
	dQ = np.zeros_like(Q)
#dH = np.zeros_like(H)

	mask = Y.copy()
	mask[mask != 0] = 1
	
	samples = [(u, i, Y[u, i]) for u in range(M) for i in range(N) if Y[u, i] > 0]

	for iter in range(n_iter):
		print(f"{iter+1:_^10} \t {n_iter:_^10}\r", end="")

		Predict = P@H@Q.T
		lossMat = Y - Predict
		loss = 0
			
		loss += np.sum(np.abs(lossMat*mask))
		dP = (-2.0 * lossMat * mask) @ Q @ H
		dQ = (-2.0 * lossMat * mask).T @ P @ H

		P, Q = P - lr*dP, Q - lr*dQ
		
		if loss <= threshold: break
		if not (iter+1)%iter_per_epoch:
			print(f"{iter+1:_^10} \t {n_iter:_^10} = \t{loss}")
	

	print(Y)
	print(P@H@Q.T)

	return P, Q, H

def MF_RealValue(Y, K, n_iter=10000, iter_per_epoch=100, lr=.001):
	# It works very well!
	M, N = Y.shape

	P = np.random.normal(loc=0, scale=0.01, size=(M, K))
	Q = np.random.normal(loc=0, scale=0.01, size=(N, K))
	dP = np.zeros_like(P)
	dQ = np.zeros_like(Q)

	mask = Y.copy()
	mask[mask != 0] = 1

	LossList = []
	for iter in range(n_iter):
		print(f"{iter+1:_^10} \t {n_iter:_^10}\r", end="")
		Predict = P@Q.T
		lossMat = Y - Predict
		loss = 0
			
		loss += np.sum((lossMat*mask)**2)
		dP = (-2.0 * lossMat * mask) @ Q
		dQ = (-2.0 * lossMat * mask).T @ P

		P, Q = P - lr*dP, Q - lr*dQ
		
		LossList.append(loss)

		if loss <= threshold: break
		if not (iter+1)%iter_per_epoch:
			print(f"{iter+1:_^10} \t {n_iter:_^10} = \t{loss}")
	

	print(Y)
	print(P@Q.T * mask)
	print(P@Q.T * (1-mask))
	print(Y - P@Q.T * mask < 0.01)

	return P, Q
	
def MF_BinaryValue(Y, K, n_iter=10000, iter_per_epoch=100, lr=.001):
	M, N = Y.shape

	P = np.random.normal(loc=0, scale=0.01, size=(M, K))
	Q = np.random.normal(loc=0, scale=0.01, size=(N, K))
	dP = np.zeros_like(P)
	dQ = np.zeros_like(Q)
	
	mask = Y.copy()
	mask[mask != 0] = 1
	
	for iter in range(n_iter):
		print(f"{iter+1:_^10} \t {n_iter:_^10}\r", end="")
		Predict = sigmoid(P@Q.T)
		dLdY = Predict - Y
		#lossMat = -Y * np.log(Predict)
		lossMat = -Y * np.log(Predict) - (1.0 - Y) * np.log(1.0 - Predict)
		loss = 0
		
		loss += np.sum(lossMat * mask)
	
		dP = (dLdY * mask) @ Q
		dQ = (dLdY * mask).T @ P

		P, Q = P - lr*dP, Q - lr*dQ
		
		if loss <= threshold: break
		if not (iter+1)%iter_per_epoch:
			print(f"{iter+1:_^10} \t {n_iter:_^10} = \t{loss}")
	
	print(Y)
	print(sigmoid(P@Q.T) * mask)
	print(sigmoid(P@Q.T) * (1-mask))
	print(Y - sigmoid(P@Q.T) * mask < 0.01)
	
	return P, Q

if __name__=="__main__":
	data_RealValue = np.loadtxt("sample.dat")
	data_BinaryValue = np.zeros_like(data_RealValue)
	data_BinaryValue[data_RealValue != 0] = 1

	data = data_BinaryValue

	MF_BinaryValue(data, int(sys.argv[1]), n_iter=20000, lr=0.01)

