#!/usr/bin/python3
#-*- coding: utf-8 -*-

import numpy as np

threshold = 1e-10
epsilon = 1e-10
np.set_printoptions(suppress=True, precision=3)

def MF_RealValue(Y, K, n_iter=10000, iter_per_epoch=100, lr=.001):
	M, N = Y.shape

	P = np.random.rand(M, K) / (M*K)
	Q = np.random.rand(N, K) / (N*K)

	LossList = []
	for iter in range(n_iter):
		Predict = P@Q.T
		lossMat = Y - Predict
		loss = 0
		for u, i in zip(*Y.nonzero()):
			loss += lossMat[u][i]**2
			LossList.append(loss)
			
			for k in range(K):
				dP = -2.0 * lossMat[u][i] * Q[i][k]
				dQ = -2.0 * lossMat[u][i] * P[u][k]

				P[u][k] -= lr*dP
				Q[i][k] -= lr*dQ

		if loss <= threshold: break
		if not (iter+1)%iter_per_epoch:
			print(f"{iter+1:_^10} \t {n_iter:_^10} = \t{loss}")
	
	return P, Q

def GMF_RealValue(Y, K, n_iter=10000, iter_per_epoch=100, lr=.001):
	M, N = Y.shape

	P = np.random.rand(M, K) / (M*K)
	Q = np.random.rand(N, K) / (N*K)
	H = np.diag(np.random.rand(K)) / K

	LossList = []
	for iter in range(n_iter):
		Predict = P@H@Q.T
		lossMat = Y - Predict
		loss = 0
		for u, i in zip(*Y.nonzero()):
			loss += lossMat[u][i]**2
			LossList.append(loss)
			
			for k in range(K):
				dP = -2.0 * lossMat[u][i] * Q[i][k] * H[k][k]
				dQ = -2.0 * lossMat[u][i] * P[u][k] * H[k][k]
				dH = -2.0 * lossMat[u][i] * P[u][k] * Q[i][k]

				P[u][k] -= lr*dP
				Q[i][k] -= lr*dQ
				H[k][k] -= lr*dH

		if not (iter+1)%iter_per_epoch:
			print(f"{iter+1:_^10} \t {n_iter:_^10} = \t{loss}")
	
	return P, Q, H

if __name__=="__main__":
	data_RealValue = np.loadtxt("sample_realvalue.dat")

	UserLatent, ItemLatent = MF_RealValue(data_RealValue, 50, iter_per_epoch=100, n_iter=1000, lr=.003)
#	UserLatent, ItemLatent, Weight = GMF_RealValue(data_RealValue, 50, n_iter=10000, lr=.005)

	mask = np.zeros_like(data_RealValue)
	mask[data_RealValue.nonzero()] = 1
	
	print(data_RealValue)
	print(np.round(UserLatent@ItemLatent.T) * mask, 0)
	print(np.round((UserLatent@ItemLatent.T) * (1-mask), 1))
	
