import numpy as np

def normalize(X, norm_type = "range"):
	""" Function performs normalization procedure on a given dataset  
	@param X - array containing vector of parameters for each observation in a row
	@norm_type - type of normalization method (range, std), where range is a default choice. """

	denominator = np.zeros(X.shape[1])
	
	if norm_type == "range":
		denominator = np.amax(X, axis = 0) - np.amin(X, axis = 0)
	elif norm_type == "std":
		denominator = np.std(X, axis = 0)
	else:
		print("Wrong type of normalization method!!")
		return()
	mean = np.mean(X, axis = 0)

	return((X - mean)/denominator)
	
#-- Testing Area --#
#X = np.genfromtxt("data.csv", delimiter = ",")
#X = X[:, 0:2]

#normalize(X, "std")


def computeCost(X, y, theta):
	""" This is a function that calculates the cost for a linear regression algorithm
	@param X - array containing vector of parameters for each observation in a row
	@param y - prediction value
	@param theta - starting values for function parameters. """
	
	#-- Initialize values
	J = 0
	m = len(y)
	
	#-- Compute the cost
	J = 1/(2*m) * sum((np.dot(X,theta) - y)**2)
	
	return(J)
	
#-- Testing area --#
#X = np.genfromtxt("data.csv", delimiter = ",")
#y = X[:,2]
#X0 = np.array(np.ones((len(y), 1), dtype = np.int)  )
#X = np.concatenate((X0, X[:,0:2]), axis = 1 )
#theta = np.array([0,0,0])

#cost = costFunction(X, y, theta)
#print(cost)

def runGradientDescent(X, y, theta, alpha, iterations):
	""" Function computes "optimal" theta parameters based on Gradient Descent algorithm
	@param X - array of input values
	@param y - array of output values
	@param theta - array of starting theta values
	@param alpha - learning rate
	@iterations - number of iterations to run gradient. """
	m = y.shape[0]
	
	#-- Create placeholders
	J_progress = np.zeros([iterations, 1])
	tmp0 = 0.0
	tmp1 = 0.0
	theta_opt = theta
	
	for i in range(0,iterations):
		tmp0 = theta_opt[0] - alpha * (1/m) * sum((np.dot(X,theta_opt) - y) * X[:,0])
		tmp1 = theta_opt[1] - alpha * (1/m) * sum((np.dot(X,theta_opt) - y) * X[:,1])
		theta_opt[0] = tmp0
		theta_opt[1] = tmp1
		
		J_progress[i] = computeCost(X, y, theta_opt)
	
	return((theta_opt, J_progress))





















