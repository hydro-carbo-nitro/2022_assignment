#!/usr/bin/python3

#Linear Fitting

import sys
import numpy as np
import matplotlib.pyplot as plt
from random import random

def gen_data(x, err, param):
    N           =   len(x)
    m           =   len(param) - 1 
    xMatrix     =   np.empty((N, m + 1))                    # x matrix with size N*(m+1)

    for i in range(N):
        for j in range(m + 1):
            xMatrix[i][j] = x[i]**j

    y           =   xMatrix @ param + err


    return xMatrix, y

def gauss_elimination(A, v):
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

    # generate data N samples with mth order
    nSample     =   int(sys.argv[1])                        # number of sample with size of N
    paramVec    =   [1, 2, 3]                               # parameter vector with size of m+1

    xVec        =   np.linspace(-2, 2, nSample)           # x vector with size N
    errVec      =   (np.random.rand(nSample) - 0.5) * 2.0   # fluctuation

    xMat, yVec  =   gen_data(xVec, errVec, paramVec)        # generate xMat and yVec

    # fitting area
    m           =   len(paramVec) - 1                       # mth order
    thetaVec    =   np.zeros(2*m+1)                         # we need 2m+1 size matrix
    phiVec      =   np.zeros(m+1)                           # we need m+1 size vector too
    
    for sample in range(nSample):
        for order in range(2*m+1):
            thetaVec[order]     +=  xVec[sample]**order / (errVec[sample]**2)
            if order <= m:
                phiVec[order]   +=  yVec[sample] * xVec[sample]**order / (errVec[sample]**2)

    thetaMat    =   np.empty((m+1, m+1))                  # theta matrix
    
    for i in range(thetaMat.shape[0]):
        for j in range(thetaMat.shape[1]):
            thetaMat[i][j]  =   thetaVec[i+j]

    coeffVec    =   gauss_elimination(thetaMat, phiVec)     # coefficient vector. thetaMat @ coeffVec = phiVec


    plt.plot(xVec, yVec, 'ro')
    plt.plot(xVec, xMat@coeffVec, '--')
    plt.grid()
    plt.show()
