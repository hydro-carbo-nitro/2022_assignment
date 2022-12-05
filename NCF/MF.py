#!/usr/bin/python3
#-*- coding: utf-8 -*-

import numpy as np

threshold = 1e-10
epsilon = 1e-10
np.set_printoptions(suppress=True, precision=3)

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

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

def MF_BinaryValue(Y, K, n_iter=10000, iter_per_epoch=100, lr=.001):
	M, N = Y.shape

	P = np.random.rand(M, K) / K
	Q = np.random.rand(N, K) / K

	LossList = []
	for iter in range(n_iter):
		Predict = sigmoid(P@Q.T)
		dLdY = Y - Predict
		#lossMat = -Y * np.log(Predict)
		lossMat = -Y * np.log(Predict) - (1.0 - Y) * np.log(1.0 - Predict)
		loss = 0
		
		posRows, posCols = np.where(Y != 0) # Positive instances.
		negRows, negCols = np.where(Y == 0) # Negative instances.
		nPos, nNeg = len(posRows), len(negRows)	# Number of positive instances and negative instances

		sampling_ratio = 4
		if nPos*sampling_ratio < nNeg:
			print(f"Sampling ratio is too high! There are ({nPos}, {nNeg}) of positive and negative instaces")
			sampling_ratio = int(nNeg / nPos)
			print(f"Change sampling ratio to be {sampling_ratio}")

		batch_size = 10
		if batch_size > nPos:
			print(f"Batch size is larger than the number of positive instances!")
			batch_size = int(nPos/2)
			print(f"Change batch size to be {batch_size}")

		posCh = np.random.randint(nPos, size=batch_size)
		negCh = np.random.randint(nNeg, size=batch_size * sampling_ratio)

		for idx in posCh:
			u, i = posRows[idx], posCols[idx]	
			loss += lossMat[u][i]
			
			for k in range(K):
				dP = dLdY[u][i] * Q[i][k]
				dQ = dLdY[u][i] * P[u][k]

				P[u][k] += lr*dP
				Q[i][k] += lr*dQ
		
		for idx in negCh:
			u, i = negRows[idx], negCols[idx]	
			loss += lossMat[u][i]
			
			for k in range(K):
				dP = dLdY[u][i] * Q[i][k]
				dQ = dLdY[u][i] * P[u][k]

				P[u][k] += lr*dP
				Q[i][k] += lr*dQ
			
		
		LossList.append(loss)
		
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
	data_BinaryValue = np.loadtxt("sample_binaryvalue.dat")

	data = data_BinaryValue

#	UserLatent, ItemLatent = MF_RealValue(data, 50, iter_per_epoch=100, n_iter=1000, lr=.003)
	UserLatent, ItemLatent = MF_BinaryValue(data, 50, iter_per_epoch=100, n_iter=10000, lr=.001)
#	UserLatent, ItemLatent, Weight = GMF_RealValue(data_RealValue, 50, n_iter=10000, lr=.005)

	mask = np.zeros_like(data)
	mask[data.nonzero()] = 1
	
	print(data)

	debug = " THIS IS PREDICT VALUE "
	blank = " "
	print(f"{debug:#^50}")
	print(np.round(sigmoid(UserLatent@ItemLatent.T) * mask, 0))
	print(f"{blank:#^50}")
	print(np.round(sigmoid(UserLatent@ItemLatent.T) * (1-mask), 1))	
	print(f"{blank:#^50}")
	
	answer0 = np.logical_and(data <= 0.1, np.round(sigmoid(UserLatent@ItemLatent.T) * (1-mask), 1) <= 0.1)
	answer1 = np.logical_and(data >= 0.9, np.round(sigmoid(UserLatent@ItemLatent.T) * mask, 1) >= 0.9)
	print(np.logical_or(answer0, answer1))


	print(UserLatent@ItemLatent.T)
