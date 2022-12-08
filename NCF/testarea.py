#!/usr/bin/python3

import numpy as np


data = np.loadtxt("./sample_realvalue.dat")

print(data)

samples = [(u, i, data[u, i]) for u in range(data.shape[0]) for i in range(data.shape[1]) if data[u, i] > 0]

print(samples)
