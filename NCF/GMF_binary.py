#!/usr/bin/python3

import numpy as np

np.set_printoptions(precision=1, suppress=True)

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

class Concatenation:
	def __init__(self, WUser, WItem):
		self.WU = WUser
		self.WI = WItem

	def forward(self, x):
		self.Set = np.int32(x)
		self.idxUser, self.idxItem = self.Set[:, 0], self.Set[:, 1]

		user = self.WU[self.idxUser]
		item = self.WI[self.idxItem]

		out = np.concatenate((user, item), axis=1)

		return out

	def backward(self, dout):
		nLatent = self.WU.shape[1]
		
		self.dWU = np.zeros_like(self.WU)
		self.dWI = np.zeros_like(self.WI)

		dWU_tmp = dout[:, :nLatent]
		dWI_tmp = dout[:, nLatent:]

		# I hope to avoid for loop...
		for sample, (u, i) in enumerate(self.Set):
			self.dWU[u] += dWU_tmp[sample]
			self.dWI[i] += dWI_tmp[sample]


class ElementWiseProduct:
	def __init__(self, WUser, WItem):
		self.WU = WUser
		self.WI = WItem

	def forward(self, x):
		self.Set = np.int32(x)
		self.idxUser, self.idxItem = self.Set[:, 0], self.Set[:, 1]

		user = self.WU[self.idxUser]
		item = self.WI[self.idxItem]

		out = user * item

		return out

	def backward(self, dout):
		self.dWU = np.zeros_like(self.WU)
		self.dWI = np.zeros_like(self.WI)

		for sample, (u, i) in enumerate(self.Set):
			self.dWU[u] += dout[sample] * self.WI[i]
			self.dWI[i] += dout[sample] * self.WU[u]
	
class SigmoidWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None	# scalar
		self.t = None	# scalar

	def forward(self, x, t):
		self.t = t.reshape(t.shape[0], -1)
		self.y = sigmoid(x)
		self.loss = -np.sum(self.y * np.log(self.y) + (1.0 - self.y) * np.log(1.0 - self.y)) / t.shape[0]

		return self.loss

	def backward(self, dout=1.0):
		dx = self.y - self.t

		return dx

class WeightDot:
	def __init__(self, W):
		self.W = W
		self.x = None

	def forward(self, x):
		self.x = x
		out = np.dot(self.x, self.W)

		return out

	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)

		return dx


class LayerNet:
	def __init__(self, dataSet, nLatent, hidSize):
		#	##########################################################################################################
		#	nUser		:	Number of users who have any nonzeros rating
		#	nItem		:	Number of items which of rating is nonzero
		#	nLatent		:	Number of latents
		#	hidSize		:	HiddenLayer size. For the simple implementation, there is only one hidden layer
		#	##########################################################################################################
		
		nUser, nItem = np.unique(dataSet[:, 0]).shape[0], np.unique(dataSet[:, 1]).shape[0]

		#	##########################################################################################################
		#	param_1		:	from input to concatenation. Just concatenation! There is no bias!
		#	param_2		:	from concatenation to hidden. Affine and Relu.
		#	param_3		:	from hidden to output. Weight dot product. It is same with Affine with zero bias
		#	##########################################################################################################

		
		self.params = {}
		self.params['WU'] = np.random.normal(loc=0, scale=0.01,  size=(nUser, nLatent))
		self.params['WI'] = np.random.normal(loc=0, scale=0.01,  size=(nItem, nLatent))
		self.params['W2'] = np.random.normal(loc=0, scale=0.01,  size=(nLatent, 1))

		self.layers = {}
		self.layers['ElementWiseProduct'] = ElementWiseProduct(self.params['WU'], self.params['WI'])
		self.layers['WeightDot'] = WeightDot(self.params['W2'])
		
		self.lastLayer = SigmoidWithLoss()

	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)

		return x

	def loss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)

	def gradient(self, x, t):
		#	Forward
		self.loss(x, t)

		#	Backward
		dout = 1.0
		dout = self.lastLayer.backward(dout)

		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)

		grads = {}
		grads['WU'] = self.layers['ElementWiseProduct'].dWU
		grads['WI'] = self.layers['ElementWiseProduct'].dWI
		grads['W2'] = self.layers['WeightDot'].dW

		return grads

	def accuracy(self, x, t):
		y = sigmoid(self.predict(x))
		
		acc = np.sum(np.fabs(y - t) <= 0.01) / t.shape[0]
		return acc

def get_samples(raw_data, ratio=4):
	M, N = raw_data.shape

	data = raw_data.copy()
	data[data != 0] = 1 # to be implicit

	# for all instances
	allSet = np.array([(u, i, data[u, i]) for u in range(M) for i in range(N)])
	
	all_x = allSet[:, :2]
	all_t = allSet[:, 2]

	# for testSet
	testSet = np.array([(u, i, data[u, i]) for u in range(M) for i in range(N) if data[u, i] > 0])
	
	test_x = testSet[:, :2]
	test_t = testSet[:, 2]
	
	# for trainSet. make some positives to be negative
	for u in range(M):
		nonzeros = np.where(data[u, :] != 0)[0] # [0] is needed
		if len(nonzeros) >= 2:
			blind = np.random.choice(nonzeros, 1, replace=False)
			data[u, blind] = 0
		
	# for positive instances
	posSet = np.array([(u, i, data[u, i]) for u in range(M) for i in range(N) if data[u, i] > 0])
	
	# for negative instances
	for u in range(M):
		negs = [i for i in range(N) if data[u, i] == 0]
		negIdx = np.random.choice(negs, ratio, replace=False)
		negSet = [(u, i, data[u, i]) for i in negIdx]

		trainSet = np.concatenate((posSet, negSet), axis=0)


	train_x = trainSet[:, :2]
	train_t = trainSet[:, 2]

	return train_x, train_t, test_x, test_t, all_x, all_t

if __name__ == "__main__":
	raw_data = np.loadtxt("sample.dat")

	train_x, train_t, test_x, test_t, all_x, all_t = get_samples(raw_data, 2)
	
	K = 4
	nIter = 50000
	lr = 0.003
	train_size = train_x.shape[0]
	batch_size = train_size // 2

	network = LayerNet(all_x, K, 64)

	for i in range(nIter):
		batch_mask = np.random.choice(train_size, batch_size, replace=False)
		batch_x = train_x[batch_mask]
		batch_t = train_t[batch_mask]
		grad = network.gradient(batch_x, batch_t)

		for key in ('WU', 'WI', 'W2'):
			network.params[key] -= lr * grad[key]

		if (i+1)%1000 == 0:
			train_loss = network.loss(train_x, train_t)
			train_acc = network.accuracy(train_x, train_t)
			test_acc = network.accuracy(test_x, test_t)
			print(f"{i+1} : loss={train_loss:4e}\t traing_acc={train_acc:4e}\t test_acc={test_acc:4e}")


	all_y = sigmoid(network.predict(all_x))

	all_y = all_y.reshape(raw_data.shape[0], raw_data.shape[1])
	all_t = all_t.reshape(raw_data.shape[0], raw_data.shape[1])
	print(all_y)
	print(all_t)


