#!/usr/bin/python3


import numpy as np

stars = [0, 1, 2, 3, 4, 5]
p = [5, 1, 1, 1, 1, 1]

data = np.random.choice(stars, size=(50, 20), p=p/np.sum(p))

np.savetxt("sample_realvalue.dat", data)
