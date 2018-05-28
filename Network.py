import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

class Network:
	def __init__(self, num_neurons, data,
		activation='sigmoid', error='cross_entropy_cost', learning_rate=0.2, epochs=40, batch_size_as_percentage=0.2):
	#	num_neurons: list containing the number of neurons in each layer - list of N integers
	#	activations: N-1 length list containing the activation function to be used for each layer (except the input layer)
	#				types of activations: 'relu', 'linear', 'step', 'sigmoid'
	#	error: the error function to be used
	#				types of error functions: 'logistic'
		self.num_layers = len(num_neurons)
		self.num_neurons = num_neurons
		self.error = cross_entropy_cost
		self.activation = sigmoid
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size_as_percentage
		self.data = data
		self.gprime = sigmoid_prime
		self.biases = [np.random.randn(x) for x in num_neurons[1:]]	#	bias vector for each layer except the input layer
		self.weights = [np.random.randn(y, x)	for x, y in zip(num_neurons[:-1], num_neurons[1:])]	#	weight matrix for each layer except the input layer 

	def feedforward(self, a):
		#	gets the activation values for the output layer of the network
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w,a) + b #weighted sum
			a = self.activation(z)	#activation value at layer i+1
		return a


	def sgd(self):
		#	stochastic gradient descent
		errors=[]
		for j in range(self.epochs):
			# if j>=100 and j< 150:
			# 	self.learning_rate = 0.25
			sum_error = 0
			batch = self.data.sample(frac=self.batch_size)
			self.xs, self.ys = get_x_y(batch)
			scaler = MinMaxScaler()
			self.xs = scaler.fit_transform(self.xs)
			self.xs = [x - 0.5 for x in self.xs]

			Delta_b = [np.zeros(b.shape) for b in self.biases]
			Delta_w = [np.zeros(w.shape) for w in self.weights]

			for x, y in zip(self.xs, self.ys):
				p_b, p_w, error = self.backprop(x, y)
				for i in range(self.num_layers-1):
					Delta_b[i] = Delta_b[i] + p_b[i]
					Delta_w[i] = Delta_w[i] + p_w[i]
				sum_error += error

			n = len(self.xs)
			D_b = [np.zeros(b.shape) for b in self.biases]
			D_w = [np.zeros(w.shape) for w in self.weights]

			for i in range(self.num_layers-1):
				D_b[i] = Delta_b[i] * 1/n
				D_w[i] = Delta_w[i] * 1/n

				#gradient descent update step
				self.biases[i] = self.biases[i] - (self.learning_rate * D_b[i])
				self.weights[i] = self.weights[i] - (self.learning_rate * D_w[i])
			# print(self.weights)
			#print(j+1, "epochs completed.")
			errors.append(sum_error)
		print("trained")
		return errors


	def backprop(self, x, y):
		#	backpropagates the error and computes partial derivatives
		zs=[]
		deltas=[]
		partial_b = [np.zeros(b.shape) for b in self.biases]	#	stores the partial derivatives w.r.t biases
		partial_w = [np.zeros(w.shape) for w in self.weights]	#	stores the partial derivatives w.r.t weights
		a_l = np.array(x)
		activations=[a_l]	#	stores the activations of each neuron by layer;	stores the input as the first layer of activations
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w,a_l) + b
			zs.append(z)
			activations.append(self.activation(z))
			a_l = activations[-1]
		error = cross_entropy_cost(y, a_l)
		deltas.insert(0, a_l - y)	#	error of the last layer
		# deltas.insert(0, cross_entropy_cost(y, a_l) * a_l)
		partial_b[-1] = deltas[0]
		partial_w[-1] = np.multiply(deltas[0].reshape(len(deltas[0]), 1),	activations[-2].reshape(1, len(activations[-2])))

		for l in range(2, self.num_layers):
			z = zs[-l]
			gprime = self.gprime(z)
			deltas.insert(0, np.dot(self.weights[-l+1].transpose(), deltas[0]) * gprime)

			partial_b[-l] = deltas[0]
			partial_w[-l] = np.multiply(deltas[0].reshape(len(deltas[0]), 1),	activations[-l-1].reshape(1, len(activations[-l-1])))
		return (partial_b, partial_w, error)


def get_x_y(data):
	xvals = data.drop('Type', axis=1)
	xs = xvals.values	#get list of numpy arrays to use as network input
	ys = []				#encode output as [1 0] if malicious, and [0 1] if benign
	for (i, row) in data.iterrows():
		y = np.array([1, 0]) if (row['Type'] == 1) else np.array([0, 1])	
		ys.append(y)
	return (xs, ys)


#ACTIVATION FUNCTIONS
def sigmoid(z):
	#	the sigmoid/logistic activation function
	return 1/(1 + np.exp(-z))
def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def relu(z):
	#	ReLu activation function
	for i in range(len(z)):
		z[i] =  z[i] if z[i] >= 0 else 0
	return z 

def step(z):
	#	the binary threshold neuron
	for i in range(len(z)):
		z[i] =  1 if z[i] >= 1 else 0
	return z 

def linear(z):
	#	linear neuron: weighted sum
	return z

#ERROR FUNCTION
def cross_entropy_cost(y, a):
	#	negative log likelihood to be minimised
	#	y - encoded labels of training data
	#	a - activations of output layer
	return np.sum(np.nan_to_num(-y*np.log(a+1e-8)-(1-y)*np.log(1-a)))	#	np.nan_to_num is used since np.log(0) = NaN