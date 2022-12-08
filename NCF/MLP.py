#!/usr/bin/python3

import numpy as np
from layers.LayerNet import *

if __name__ == "__main__":
	raw_data = np.loadtxt("sample_realvalue.dat")
	
	M, N = raw_data.shape
	K = 5
	
	samples = np.array([(u, i, raw_data[u, i]) for u in range(M) for i in range(N) if raw_data[u, i] > 0])

	x = samples[:, 0:2]
	t = samples[:, 2]

	Embedding_User = np.random.normal(loc=0, scale=0.01, size=(M*K))
	Embedding_Item = np.random.normal(loc=0, scale=0.01, size=(N*K))

	user_latent = Embedding_User.flatten()
	item_latent = Embedding_item.flatten()

	
	"""
	grad = network.gradient(x_batch, t_batch)

	for key in network.params.keys():
		network.params[key] -= learning_rate * grad[key]
	
	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)
	"""
