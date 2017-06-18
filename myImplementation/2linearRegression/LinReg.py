from statistics import mean
import numpy as np

class LinReg:
	
	#-- Condstructor
	def __init__(self, xs, ys):
		self.x = xs
		self.y = ys
		self.slope = ( ((mean(self.x) * mean(self.y)) - mean(self.x*self.y)) /
					   (mean(self.x)**2 - mean(self.x**2)))
		self.intercept = mean(self.y) - self.slope * mean(self.x)
		
		
	
	#-- Methods
	def getSlope(self):
		return(self.slope)
	
	def getIntercept(self):
		return(self.intercept)
	
	def predict(self,x):
		return((self.slope*x) + self.intercept)
	
	def calculateTSS(self):
		return(sum([(y_i - mean(self.y))**2 for y_i in self.y]))
	
	def calculateRSS(self):
		return(sum([(self.y[i] - self.predict(self.x[i]))**2 for i in range(len(self.x))]))
	
	def getR2(self):
		return( 1 - self.calculateRSS()/self.calculateTSS())
	
	
	