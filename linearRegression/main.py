import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import functions as fun


def main():
	#-- 1. Load the dataset
	X = np.genfromtxt("ex1data1.txt", delimiter = ",")
	y = X[:,1]
	X0 = np.array(np.ones((len(y), 1), dtype = np.float))
	X = np.concatenate((X0, X[:,0:1]), axis = 1 )
	theta = np.array(np.zeros(X.shape[1]))

	#-- 2. Print scatterplot of the data X vs y
	fig1 = plt.figure()
	plt.plot(X[:,1], y, 'ro')
	plt.title("House costs in US.")
	plt.xlabel("Population of City in 10,000s")
	plt.ylabel("Profit in $10,000s")


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
	plt.plot(X[:,1], np.dot(X,theta_opt), 'k-')
	plt.show()

	theta0_3D = np.arange(-10,10,0.2)
	theta1_3D = np.arange(-1,4,0.05)
	J_3D = np.zeros([theta0_3D.shape[0], theta1_3D.shape[0]])

	for i in range(0, theta0_3D.shape[0]):
		for j in range(0, theta1_3D.shape[0]):
			J_3D[i,j] = fun.computeCost(X, y, [theta0_3D[i], theta1_3D[j]])
	theta0_3D, theta1_3D = np.meshgrid(theta0_3D, theta1_3D)

	#-- Surface plot
	fig2 = plt.figure()
	ax = fig2.gca(projection = '3d')
	surf = ax.plot_surface(X = theta0_3D, Y = theta1_3D, Z = J_3D, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	fig2.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()

	#-- Countour plot
	X = theta0_3D
	Y = theta1_3D
	Z = J_3D

	plt.figure()
	CS = plt.contour(Y, X, Z, )
	plt.clabel(CS, inline=1, fontsize=10)
	plt.title('Contour plot')
	plt.plot([theta_opt[1]], [theta_opt[0]], 'rx', markersize = 10.0)
	plt.show()
	

main()




