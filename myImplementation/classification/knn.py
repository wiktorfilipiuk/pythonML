import sys
import numpy as np
import warnings

from collections import Counter

sys.path.append('../myFunctions')
import distance as dist

class Knn:
	data = []
	K = 0
	#-- Constructor of the class
	def __init__(self, data=[], K=0):
		self.data = data
		self.K = K
		
	def train_method(self):
		return(0)
	
	def predict(self, new_data, retConfidence=False):
		distances = []
		for group in self.data:
			for features in self.data[group]:
				distances.append([dist.euclideanDistance(features, new_data), group])
		distances.sort()
		votes = [item[1] for item in distances[:self.K]]
		vote_result = Counter(votes).most_common(1)[0][0]
		vote_confidence = Counter(votes).most_common(1)[0][1] / self.K
		
		if retConfidence:
			return(vote_result, vote_confidence)
		else:
			return(vote_result)



