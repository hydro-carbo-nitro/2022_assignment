#!/usr/bin/python3


import numpy as np

stars = [0, 1, 2, 3, 4, 5]
p = [7, 1, 1, 1, 1, 1]

data = np.random.choice(stars, size=(60, 30), p=p/np.sum(p))
"""
data = np.array([
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
])
"""
np.savetxt("sample_realvalue.dat", data)
