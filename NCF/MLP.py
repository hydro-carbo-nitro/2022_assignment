#!/usr/bin/python3

import numpy as np

def MSE(y, t):
	return (y - t)**2

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
	def __init__(self, WUser, WItem):
		self.WU = WUser
		self.WI = WItem

	def forward(self, x):
		self.idxUser, self.idxItem = np.int32(x[:, 0]), np.int32(x[:, 1])
		user = self.WU[self.idxUser]
		item = self.WI[self.idxItem]

		print(user)

		out = np.concatenate((user, item), axis=1)
	
		return out

	def backward(self, dout):
		nLatent = self.WU.shape[1]
		
		self.dWU = np.zeros_like(self.WU)
		self.dWI = np.zeros_like(self.WI)

		dWU_tmp = dout[:, :nLatent]
		dWI_tmp = dout[:, nLatent:]
		
		# I hope to avoid for loop...
		for u, i in zip(self.idxUser, self.idxItem):
			self.dWU += dWU_tmp
			self.dWI += dWI_tmp
	
class IdentityWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None	# Scalar
		self.t = None	# Scalar

	def forward(self, x, t):
		self.t = t
		self.y = x	# Identity function
		print(t.shape)
		print(x.shape)
		self.loss = (self.y - self.t) ** 2

		return self.loss

	def backward(self, dout=1):
		dx = 2.0 * (self.y - self.t)

		return dx

class LayerNet:
	def __init__(self, nSet, nLatent, hidSize):
		#	##########################################################################################################
		#	nSet		:	Number of pairs which of rating is nonzero
		#	nUser		:	Number of users who have any nonzeros rating
		#	nItem		:	Number of items which of rating is nonzero
		#	nLatent		:	Number of latents
		#
		#	inSize		:	InputLayer size. It is always 2. User index and item index
		#	concSize	:	ConcatenationLayer size. It is 2K. K is the number of latent factor.
		#	hidSize		:	HiddenLayer size. For the simple implementation, there is only one hidden layer
		#	outSize		:	OutputLayer size. It is always 1. Because there is only one answer for (user, item) pair
		#	##########################################################################################################

		inSize = 2
		concSize = 2 * nLatent
		outSize = 1
		
		nUser, nItem = np.unique(nSet[:, 0]).shape[0], np.unique(nSet[:, 1]).shape[0]

		#	##########################################################################################################
		#	param_1		:	from input to concatenation. Just concatenation! There is no bias!
		#	param_2		:	from concatenation to hidden. Affine and Relu.
		#	param_3		:	from hidden to output. Weight dot product. It is same with Affine with zero bias
		#	##########################################################################################################

		
		self.params = {}
		self.params['WU'] = np.random.normal(loc=0, scale=0.01,  size=(nUser, nLatent))
		self.params['WI'] = np.random.normal(loc=0, scale=0.01,  size=(nItem, nLatent))
		self.params['W2'] = np.random.normal(loc=0, scale=0.01,  size=(concSize, hidSize))
		self.params['b2'] = np.zeros(hidSize)
		self.params['W3'] = np.random.normal(loc=0, scale=0.01,  size=(hidSize, outSize))

		self.layers = {}
		self.layers['Concatenation'] = Concatenation(self.params['WU'], self.params['WI'])			# Not Yet
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
		self.layers['Relu2'] = Relu()
		self.layers['WeightDot'] = Affine(self.params['W3'], np.zeros(outSize))
		
		self.lastLayer = IdentityWithLoss()

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
		grads['WU'] = self.layers['Concatenation'].dWU
		grads['WI'] = self.layers['Concatenation'].dWI
		grads['W2'] = self.layers['Affine2'].dW
		grads['b2'] = self.layers['Affine2'].db
		grads['W3'] = self.layers['WeightDot'].dW

		return grads

if __name__ == "__main__":
	raw_data = np.loadtxt("sample_realvalue.dat")
	
	M, N = raw_data.shape
	K = 2
	nIter = 100
	lr = 0.001
	
	samples = np.array([(u, i, raw_data[u, i]) for u in range(M) for i in range(N) if raw_data[u, i] > 0])

	x = samples[:, 0:2]
	t = samples[:, 2]


	network = LayerNet(x, K, 3)

	for key in network.params.keys():
		print(network.params[key].shape)

	for i in range(nIter):
		grad = network.gradient(x, t)

		for key in ('WU', 'WI', 'W2', 'b2', 'W3'):
			network.params[key] -= lr * grad[key]

		loss = network.loss(x, t)
		print(f"{nIter} : {loss}")
	

