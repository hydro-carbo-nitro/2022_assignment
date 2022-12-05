#!/usr/bin/python3

import numpy as np

A = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 1, 1, 0]])

u, i = np.where(A != 0)
print(u, i)


rows, cols = np.where(A == 0)
u = np.concatenate([u, rows], 0)
i = np.concatenate([i, cols], 0)
print(u, i)

"""
print(np.nonzero(A))
rows, cols = np.where(A != 0)
print(rows, cols)

choice = np.random.randint(len(rows), size=3)

for idx in choice:
	u, i = rows[idx], cols[idx]

	A[u][i] = 10

print(A)
"""
