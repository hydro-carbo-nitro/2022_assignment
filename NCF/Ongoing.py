#!/usr/bin/python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

threshold = 1e-10
epsilon = 1e-10
np.set_printoptions(suppress=True, precision=3)

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))
"""
def GeneralMatrixFactorization(Y, K, n_iter=3000, iter_per_epoch=100, lr=.001):
	M, N = data.shape

	P = np.random.rand(M, K) / np.sqrt(K)
	Q = np.random.rand(N, K) / np.sqrt(K)
	H = np.diag(np.random.rand(K)) / np.sqrt(K)
	#H = np.diag(np.ones(K)) / np.sqrt(K)
	
	LossList = []
	for iter in range(n_iter):
		dp = np.zeros_like(P)
		dq = np.zeros_like(Q)
		dh = np.zeros_like(H)
	
		Y_predict = sigmoid(P@H@Q.T)
		Loss = -np.sum(Y * np.log(Y_predict + epsilon) + (1.0 - Y) * np.log((1.0 - Y_predict + epsilon)))
		LossList.append(Loss)

		dLdY = -Y/Y_predict + (1.0 - Y)/(1.0 - Y_predict)
	
		for u in range(M):
			for i in range(N):
				for k in range(K):
					dp[u][k] += dLdY[u][i] * Y_predict[u][i] * (1.0 - Y_predict[u][i]) * H[k][k] * Q[i][k]
					dq[i][k] += dLdY[u][i] * Y_predict[u][i] * (1.0 - Y_predict[u][i]) * H[k][k] * P[u][k]
					dh[k][k] += dLdY[u][i] * Y_predict[u][i] * (1.0 - Y_predict[u][i]) * P[u][k] * Q[i][k]

		P, Q, H = P - lr*dp, Q - lr*dq, H - lr*dh
		#P, Q = P - lr*dp, Q - lr*dq

		if not (iter+1)%100:
			print(f"{iter+1} \t {n_iter} = \t{Loss}")

	return P, Q, H

def MF_RealValue(Y, K, n_iter=10000, iter_per_epoch=100, lr=.001):
	M, N = Y.shape

	P = np.random.rand(M, K) / (M*K)
	Q = np.random.rand(N, K) / (N*K)

	dP = np.zeros_like(P)
	dQ = np.zeros_like(Q)

	LossList = []
	for iter in range(n_iter):
		Predict = P@Q.T
		loss = np.sum((Y - Predict)**2)
		LossList.append(loss)
		if loss <= threshold: break

		dP = -2.0 * (Y - Predict) @ Q
		dQ = -2.0 * (Y - Predict).T @ P

		P -= lr*dP
		Q -= lr*dQ

		if not (iter+1)%100:
			print(f"{iter+1:_^10} \t {n_iter:_^10} = \t{loss}")
	
	return P, Q
"""
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

		if not (iter+1)%iter_per_epoch:
			print(f"{iter+1:_^10} \t {n_iter:_^10} = \t{loss}")
	
	return P, Q

def MF_BinaryValue(Y, K, n_iter=10000, iter_per_epoch=100, lr=.001):
	M, N = Y.shape

	P = np.random.rand(M, K) / (M*K)
	Q = np.random.rand(N, K) / (N*K)

	LossList = []
	for iter in range(n_iter):
		Predict = sigmoid(P@Q.T)
		dLdY = -Y/(Predict+epsilon) + (1.0 - Y)/(1.0 - Predict + epsilon)
		loss = 0
		for u, i in zip(*Y.nonzero()):
			loss += -Y[u][i] * np.log(Predict[u][i])
			LossList.append(loss)
			
			for k in range(K):
				dP = dLdY[u][i] * Q[i][k]
				dQ = dLdY[u][i] * P[u][k]

				P[u][k] -= lr*dP
				Q[i][k] -= lr*dQ

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

"""
def GMF_RealValue(Y, K, n_iter=10000, iter_per_epoch=100, lr=.001):
	M, N = Y.shape

	P = np.random.rand(M, K) / M
	Q = np.random.rand(N, K) / N
	H = np.diag(np.random.rand(K)) / K

	dP = np.zeros_like(P)
	dQ = np.zeros_like(Q)
	dH = np.zeros_like(H)

	LossList = []
	for iter in range(n_iter):
		Predict = P@H@Q.T
		loss = np.sum((Y - Predict)**2)
		LossList.append(loss)
		if loss <= threshold: break

		dP = -2.0 * (Y - Predict) @ Q @ H
		dQ = -2.0 * (Y - Predict).T @ P @ H
		dH = -2.0 * Q.T @ (Y - Predict).T @ P

		P -= lr*dP
		Q -= lr*dQ
		H -= lr*dH

		if not (iter+1)%100:
			print(f"{iter+1:_^10} \t {n_iter:_^10} = \t{loss}\t[{np.min(Predict)}, {np.max(Predict)}]")
	
	return P, Q, H
"""

if __name__=="__main__":
	data_RealValue = np.loadtxt("sample_realvalue.dat")
	data_BinaryValue = np.loadtxt("sample_binaryvalue.dat")

	UserLatent, ItemLatent = MF_BinaryValue(data_BinaryValue, 50, iter_per_epoch=100, n_iter=1000, lr=.003)
#	UserLatent, ItemLatent = MF_RealValue(data_RealValue, 50, iter_per_epoch=100, n_iter=1000, lr=.003)
#	UserLatent, ItemLatent, Weight = GMF_RealValue(data_RealValue, 50, n_iter=10000, lr=.005)

	mask = np.zeros_like(data_BinaryValue)
	mask[data_BinaryValue.nonzero()] = 1
	
	print(data_BinaryValue)
	print(np.round(sigmoid(UserLatent@ItemLatent.T)), 0)
	print(np.round(sigmoid(UserLatent@ItemLatent.T) * mask), 0)
	print(np.round(sigmoid(UserLatent@ItemLatent.T) * (1-mask)), 1)
	
