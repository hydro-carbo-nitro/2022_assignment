#!/usr/bin/python3

import numpy as np
import sys
np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)

t = np.loadtxt("sample2.dat")
y = np.loadtxt("LearnedGMF.dat")

y[t != 0] = 0
print(t[0].T)
print(y[0].T)
