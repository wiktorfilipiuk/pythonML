import LinReg as lr
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import random

style.use('fivethirtyeight')

#-- Testing-Purpose-Functions
def createDataset(hm, variance, step=2, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	
	xs = [i for i in range(len(ys))]
	
	return(np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64))


def main():
	#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
	#ys = np.array([5,4,6,5,6,7], dtype=np.float64)
	
	xs, ys = createDataset(40, 20, 2, correlation='neg')
	
	lm = lr.LinReg(xs, ys)
	a = lm.getSlope()
	b = lm.getIntercept()
	print("Equation: Y = {0:1,.2f}*X + {1:1,.3f}".format(b, a))
	
	regression_line = [(a*x)+b for x in xs]
	
	predict_x = 5.5
	predict_y = lm.predict(predict_x)
	
	print("R-squared: {0:1,.3f}".format(lm.getR2()))
	
	plt.scatter(xs, ys)
	plt.plot(xs, regression_line)
	plt.scatter(predict_x, predict_y, color='red', s=100)
	plt.show()






main()