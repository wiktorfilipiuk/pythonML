import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

#-- Load dataset
df = pd.read_csv('breastCancer.txt')

#-- Handle missing data
df.replace('?', -99999, inplace = True)

#-- Transform the data
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_data = np.array([10,2,1,1,1,2,3,2,1])
example_data = example_data.reshape(1,-1)
prediction = clf.predict(example_data)
print(prediction)




