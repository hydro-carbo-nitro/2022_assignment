#!/usr/bin/python3

#Linear Fitting

import numpy as np
import matplotlib.pyplot as plt
from random import random
	
class LinearSquareFit:
	def __init__(self, data, order):
		self.xVec	=	data[:, 0]								# xVector with N*1
		self.yVec	=	data[:, 1]								# yVector with N*1
		self.errVec	=	data[:, 2]								# errorVector with N*1

		self.N		=	len(data)
		self.m		=	order + 1								# linear equation's order. 0th order ~ mth order

		# xMat		:	N*m matrix. N of samples, (m-1) of orders
		# thetaMat	;	theta is made up x^m
		# phiVec	:	phiVec is made up x^(m-1) * y
		# coeffVec	:	coefficient of x^m
		# thetaMat@coeffVec=phiVec
		self.xMat, self.thetaMat, self.phiVec	=	self.init_Fitting()								
		self.coeffVec	=	self.GaussElimination(self.thetaMat, self.phiVec)
		
	def init_Fitting(self):
		# Is there better way to generate matrix?

		# Generate x Matrix
		xMatrix = np.empty((self.N, self.m)) 
		for i in range(self.N):
			for j in range(self.m):
				xMatrix[i][j]	=	self.xVec[i]**j

		
		# Generate thetaMatrix and phiVector
		thetaVector	=	np.zeros(2*self.m - 1)			# x^idx = thetaVec[idx]
		phiVector	=	np.zeros(self.m)				# x^(idx-1) * y = phiVec[idx]

		for i in range(self.N):							# for all samples
			for order in range(2*self.m - 1):			# for all orders
				thetaVector[order]		+=	self.xVec[i]**order / (self.errVec[i]**2)
				if order <= self.m-1:
					phiVector[order]	+=	self.yVec[i] * self.xVec[i]**order / (self.errVec[i]**2)

		thetaMatrix	=	np.empty((self.m, self.m))
		
		for i in range(self.m):
			for j in range(self.m):
				thetaMatrix[i][j]		=	thetaVector[i+j]

		return xMatrix, thetaMatrix, phiVector


	def GaussElimination(self, A, v):
		#Ax = v
		N = len(v)
		pivot_index =   0

		for m in range(N):
			# Partial pivoting
			pivot_max       =   abs(A[m, m])
			pivot_point     =   m

			for i in range(m + 1, N):
				pivot_temp      =   abs(A[i, m])
				if pivot_temp   >   pivot_max:
					pivot_index, pivot_max  =   i, pivot_temp

				if pivot_index  !=  m:
					for i in range(N):
						A[m, i], A[pivot_index, i]  =   A[pivot_index, i], A[m, i]
					v[m], v[pivot_index]    =   v[pivot_index], v[m]

			# Divide by the diagonal element
			div         =   A[m, m]
			A[m, m:]    /=  div
			v[m]        /=  div
			
			# Subtract for the lower rows
			for i in range(m+1, N):
				mult        =   A[i, m]
				A[i, m:]    -=  mult * A[m, m:]
				v[i]        -=  mult * v[m]

		x   =   np.empty(N, float)

		# Back subtraction
		for m in range(N-1, -1, -1):
			x[m]    =   v[m]
			for i in range(m+1, N):
				x[m]    -=  A[m, i]*x[i]

		return x

if __name__ == "__main__":
	# load data. [col1:x, col2:y, col3:err]

	rawData		=	np.loadtxt("./data/func_data.txt")
	fitData		=	LinearSquareFit(rawData, 10)

	xList		=	fitData.xVec
	yList		=	fitData.xMat @ fitData.coeffVec	

	plt.plot(rawData[:, 0], rawData[:, 1], 'ro')
	plt.plot(xList, yList, '--')
	plt.show()
