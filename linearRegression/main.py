import numpy as np
import matplotlib.pyplot as plt
import functions as fun

def main():
	#-- 1. Load the dataset
	X = np.genfromtxt("ex1data1.txt", delimiter = ",")
	y = X[:,1]
	X0 = np.array(np.ones((len(y), 1), dtype = np.float))
	X = np.concatenate((X0, X[:,0:1]), axis = 1 )
	theta = np.array(np.zeros(X.shape[1]))
	
	#-- 2. Print scatterplot of the data X vs y
	plt.plot(X[:,1], y, 'ro')
	plt.title("House costs in US.")
	plt.xlabel("Population of City in 10,000s")
	plt.ylabel("Profit in $10,000s")
	plt.show()
	
	#-- 3. Compute the cost for predefined values of theta
	cost = fun.computeCost(X, y, theta)
	print(cost)
	theta2 = np.array([-1,2])
	cost2 = fun.computeCost(X, y, theta2)
	print(cost2)

	#-- 4. Perform Gradient Descent operation
	alpha = 0.01
	iterations = 1500
	(theta_opt, J_progress) = fun.runGradientDescent(X, y, theta, alpha, iterations)
	print(theta_opt)
	
	#-- 5. Visualize the results
	plt.plot(X[:,1], y, 'k-')
	plt.show()
	

main()