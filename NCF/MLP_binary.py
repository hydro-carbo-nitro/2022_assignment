#!/usr/bin/python3

import numpy as np

np.set_printoptions(precision=1, suppress=True)

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

class Relu:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0
		
		return out
	
	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout

		return dx

class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None

		self.dW = None
		self.db = None
		
	def forward(self, x):
		self.x = x
		out = np.dot(self.x, self.W) + self.b

		return out

	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis=0)

		return dx

class Concatenation:
	def __init__(self, P, Q):
		self.P = P
		self.Q = Q

	def forward(self, x):
		self.Set = np.int32(x)
		self.idxUser, self.idxItem = self.Set[:, 0], self.Set[:, 1]

		user = self.P[self.idxUser]
		item = self.Q[self.idxItem]

		out = np.concatenate((user, item), axis=1)
		return out

	def backward(self, dout):
		nLatent = self.P.shape[1]
		
		self.dP = np.zeros_like(self.P)
		self.dQ = np.zeros_like(self.Q)

		dP_tmp = dout[:, :nLatent]
		dQ_tmp = dout[:, nLatent:]

		# I hope to avoid for loop...
		for sample, (u, i) in enumerate(self.Set):
			self.dP[u] += dP_tmp[sample]
			self.dQ[i] += dQ_tmp[sample]

class SigmoidWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None	# Scalar
		self.t = None	# Scalar

	def forward(self, x, t):
		self.t = t.reshape(t.shape[0], -1)
		self.y = sigmoid(x)
		self.loss = -np.sum(self.y * np.log(self.y) + (1.0 - self.y) * np.log(1.0 - self.y)) / t.shape[0]

		return self.loss

	def backward(self, dout=1.0):
		dx = self.y - self.t

		return dx

class WeightDot:
	def __init__(self, H):
		self.H = H
		self.x = None

	def forward(self, x):
		self.x = x
		out = np.dot(self.x, self.H)

		return out

	def backward(self, dout):
		dx = np.dot(dout, self.H.T)
		self.dH = np.dot(self.x.T, dout)

		return dx


class LayerNet:
	def __init__(self, nSet, nLatent, hidSize):
		#	##########################################################################################################
		#	nSet		:	Number of pairs which of rating is nonzero
		#	nUser		:	Number of users who have any nonzeros rating
		#	nItem		:	Number of items which of rating is nonzero
		#	nLatent		:	Number of latents
		#	hidSize		:	HiddenLayer size. For the simple implementation, there is only one hidden layer
		#	##########################################################################################################
		
		nUser, nItem = np.unique(nSet[:, 0]).shape[0], np.unique(nSet[:, 1]).shape[0]

		#	##########################################################################################################
		#	param_1		:	from input to concatenation. Just concatenation! There is no bias!
		#	param_2		:	from concatenation to hidden. Affine and Relu.
		#	param_3		:	from hidden to output. Weight dot product. It is same with Affine with zero bias
		#	##########################################################################################################

		
		self.params = {}
		self.params['P'] = np.random.normal(loc=0, scale=0.01,  size=(nUser, nLatent))
		self.params['Q'] = np.random.normal(loc=0, scale=0.01,  size=(nItem, nLatent))
		self.params['W1'] = np.random.normal(loc=0, scale=0.01,  size=(2*nLatent, hidSize))
		self.params['b1'] = np.zeros(hidSize)
		self.params['H'] = np.random.normal(loc=0, scale=0.01,  size=(hidSize, 1))

		self.layers = {}
		self.layers['Concatenation'] = Concatenation(self.params['P'], self.params['Q'])
		self.layers['Affine2'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Relu2'] = Relu()
		self.layers['WeightDot'] = WeightDot(self.params['H'])
		
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
		grads['P'] = self.layers['Concatenation'].dP
		grads['Q'] = self.layers['Concatenation'].dQ
		grads['W1'] = self.layers['Affine2'].dW
		grads['b1'] = self.layers['Affine2'].db
		grads['H'] = self.layers['WeightDot'].dH

		return grads

	def accuracy(self, x, t):
		y = sigmoid(self.predict(x))
		t = t.reshape(t.shape[0], -1)

		acc = np.sum(np.fabs(y - t) <= 0.1) / t.shape[0]
		return acc

def getTrainData(raw_data, ratio=4):
	M, N = raw_data.shape

	data = raw_data.copy()
	
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

	return train_x, train_t


def getData(raw_data):
	M, N = raw_data.shape

	# for all instances
	allSet = np.array([(u, i, raw_data[u, i]) for u in range(M) for i in range(N)])
	
	all_x = allSet[:, :2]
	all_t = allSet[:, 2]

	# for testSet. Only positive instances
	testSet = np.array([(u, i, raw_data[u, i]) for u in range(M) for i in range(N) if raw_data[u, i] > 0])
	
	test_x = testSet[:, :2]
	test_t = testSet[:, 2]

	return all_x, all_t, test_x, test_t

if __name__ == "__main__":
	raw_data = np.loadtxt("sample2.dat")
	raw_data[raw_data != 0] = 1 # to be implicit
	
	all_x, all_t, test_x, test_t = getData(raw_data)
	
	K = 20
	nIter = 30000
	lr = 0.001

	network = LayerNet(all_x, K, 128)

	for i in range(nIter):
		train_x, train_t = getTrainData(raw_data, 3)

		train_size = train_x.shape[0]
		batch_size = 250
		
		batch_mask = np.random.choice(train_size, batch_size, replace=False)
		batch_x = train_x[batch_mask]
		batch_t = train_t[batch_mask]
		grad = network.gradient(batch_x, batch_t)

		for key in ('P', 'Q', 'W1', 'b1', 'H'):
			network.params[key] -= lr * grad[key]

		if (i+1)%1000 == 0:
			train_loss = network.loss(train_x, train_t)
			train_acc = network.accuracy(train_x, train_t)
			test_acc = network.accuracy(test_x, test_t)
			print(f"Epoch{(i+1)//1000}\t loss={train_loss:.4e}\t train_acc={train_acc:.2%}\t test_acc={test_acc:.2%}")


	all_y = sigmoid(network.predict(all_x))

	all_y = all_y.reshape(raw_data.shape[0], raw_data.shape[1])
	all_t = all_t.reshape(raw_data.shape[0], raw_data.shape[1])
	
	np.savetxt("LearnedMLPBinary.dat", all_y)


