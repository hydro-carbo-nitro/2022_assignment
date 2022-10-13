#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

path	=	"./data/"

def sin_data(xVec):
	return np.sin(xVec)

def func_data(xVec):
	return 1 + 2*xVec + 3*xVec*xVec

if __name__ == "__main__":
	xlim	=	(-3.0, 3.0)
	nSample	=	101

	xVec	=	np.linspace(xlim[0], xlim[1], nSample)
	errVec	=	(np.random.rand(nSample) - 0.5) * 2.0


	np.savetxt("./data/sin_data.txt", np.stack([xVec, sin_data(xVec)+errVec, errVec], 1), fmt='%.4f')
	np.savetxt("./data/func_data.txt", np.stack([xVec, func_data(xVec)+errVec, errVec], 1), fmt='%.4f')
