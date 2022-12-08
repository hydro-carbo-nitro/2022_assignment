#!/usr/bin/python3
# coding: utf-8
import numpy as np
from .Layers import *

class MultiLayerNet:
	def __init__(self, size_list):
		self.params, self.layers = {}, {}
		self.W_keys, self.b_keys, self.Affine_keys = [], [], []
		for idx in range(1, len(size_list)):
			_W, _b, _Affine, _Activation = "W"+str(idx), "b"+str(idx), "Affine"+str(idx), "Activation"+str(idx)
			self.W_keys.append(_W)
			self.b_keys.append(_b)
			self.Affine_keys.append(_Affine)

			self.params[_W] = 0.01 * np.random.randn(size_list[idx-1], size_list[idx])
			self.params[_b] = np.zeros(size_list[idx])
			self.layers[_Affine] = Affine(self.params[_W], self.params[_b])
			if idx < len(size_list) - 1: self.layers[_Activation] = Relu()

		self.lastLayer = AddLayer()


	def predict(self, x):
		for layer in self.layers.values(): x = layer.forward(x)

		return x

	def loss(self, x, t):
		y = self.predict(x)

		return self.lastLayer.forward(y, t)


	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)

		if t.ndim != 1 : t = np.argmax(t, axis=1)

		accuracy = np.sum(y == t) / float(x.shape[0])

		return accuracy

	def gradient(self, x, t):
        # forward
		self.loss(x, t)

        # backward
		dout = 1
		dout = self.lastLayer.backward(dout)

		layers = list(self.layers.values())
		layers.reverse()

		for layer in layers:
			dout = layer.backward(dout)

        # 결과 저장
		grads = {}
		for _W, _b, _Affine in zip(self.W_keys, self.b_keys, self.Affine_keys):
			grads[_W], grads[_b] = self.layers[_Affine].dW, self.layers[_Affine].db

		return grads

