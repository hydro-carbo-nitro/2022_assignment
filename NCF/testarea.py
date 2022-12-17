#!/usr/bin/python3

import numpy as np


data = np.loadtxt("./sample_realvalue2.dat")

print(data.shape)
"""
samples = [(u, i, data[u, i]) for u in range(data.shape[0]) for i in range(data.shape[1]) if data[u, i] > 0]

print(samples)
"""
all = np.array([(u, i, data[u, i]) for u in range(data.shape[0]) for i in range(data.shape[1])])


print(all[:, :2])
