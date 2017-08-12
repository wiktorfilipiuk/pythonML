import matplotlib.pyplot as plt
from matplotlib import style
import sys
import pandas as pd
import random

sys.path.append('./myFunctions')
sys.path.append('./classification')
import distance as dist
import knn

style.use('fivethirtyeight')

def main():
	dataset = {'g':[[1,2],[2,3],[3,1]],
			   'b':[[6,5],[7,7],[8,6]]}
	new_data = [3,5]
	#==============================================================================
	#-- Test knn algorithm you implemented
	#==============================================================================
	model = knn.Knn(data=dataset, K=3)
	prediction = model.predict(new_data)
	
	[[plt.scatter(j[0], j[1], s=70, color=i) for j in dataset[i]] for i in dataset]
	plt.scatter(new_data[0], new_data[1], s=200, color=prediction)
	#plt.show()
	
	#==============================================================================
	#-- Investigate breat-cancer dataset (own implementation)
	#==============================================================================
	#- Load & Transform the data
	df = pd.read_csv("../dataScience_applications/breastCancer.txt")
	df.replace('?', -99999, inplace = True)
	df.drop(['id'], 1, inplace = True)
	
	accuracies = []
	for j in range(25):
		full_data = df.astype(float).values.tolist()
		random.shuffle(full_data)

		test_size = 0.2
		train_set = {2: [], 4: []}
		test_set = {2: [], 4: []}

		train_data = full_data[:-int(test_size*len(full_data))]
		test_data = full_data[-int(test_size*len(full_data)):]

		for i in train_data:
			train_set[i[-1]].append(i[:-1])
		for i in test_data:
			test_set[i[-1]].append(i[:-1])

		correct = 0
		total = 0
		#-- Train the model
		model = knn.Knn(data=train_set,K=5)
		#-- Perform classifiaction of the test set
		model = knn.Knn(data=train_set,K=5)
		for group in test_set:
			for data in test_set[group]:
				vote, confidence = model.predict(data, retConfidence=True)
				if group == vote:
					correct += 1
				total += 1
		accuracies.append(correct/total)
		#print("Correct: {0}, Total: {1}".format(correct, total))
		#print("Accuracy: {0}".format(correct/total))
	
	print("Average Accuracy: {}".format(sum(accuracies)/len(accuracies)))
	
main()

















