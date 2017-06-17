import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import functions as fun

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold

verbose = False

#-- 1. Load the dataset
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

if verbose:
	print(data_train.sample(3))

#-- 2. Visualize data - vol. 1
sns.barplot(x = "Embarked", y = "Survived", hue = "Sex", data = data_train)
if verbose:
	plt.show()

sns.pointplot(x = "Pclass", y = "Survived", hue = "Sex", data = data_train,
			  palette = {"male": "blue", "female": "red"},
			 markers = ["*", "o"], linestyles = ["-", "--"])
if verbose:
	plt.show()

#-- 3. Features' transformation
data_train = fun.simplify_ages(data_train)
data_train = fun.simplify_cabins(data_train)
data_train = fun.simplify_fares(data_train)
data_train = fun.format_name(data_train)
data_train = fun.drop_features(data_train)

data_test = fun.simplify_ages(data_test)
data_test = fun.simplify_cabins(data_test)
data_test = fun.simplify_fares(data_test)
data_test = fun.format_name(data_test)
data_test = fun.drop_features(data_test)


if verbose:
	print(data_train.head())

#-- 4. Visualize data - vol. 2
sns.barplot(x = "Age", y = "Survived", hue = "Sex", data = data_train)
if verbose:
	plt.show()

sns.barplot(x = "Cabin", y = "Survived", hue = "Sex", data = data_train)
if verbose:
	plt.show()

sns.barplot(x = "Fare", y = "Survived", hue = "Sex", data = data_train)
if verbose:
	plt.show

data_train, data_test = fun.encode_features(data_train, data_test)
if verbose:
	print(data_train.head())

#-- 5. Splitting the data
X_all = data_train.drop(['Survived', 'PassengerId'], axis = 1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = num_test, random_state = 23)

#-- 6. Fitting and Tuning the Algorithm
clf = RandomForestClassifier()
parameters = {'n_estimators': [4, 6, 9],
			  'max_features': ['log2', 'sqrt', 'auto'],
			  'criterion': ['entropy', 'gini'],
			  'max_depth': [2, 3, 5, 10],
			  'min_samples_split': [2, 3, 5],
			  'min_samples_leaf': [1, 5, 8]
			 }
acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring = acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

clf = grid_obj.best_estimator_
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

#-- 7. KFold Validation
def run_kfold(clf):
	kf = KFold(891, n_folds = 10)
	outcomes = []
	fold = 0
	for train_index, test_index in kf:
		fold += 1
		X_train, X_test = X_all.values[train_index], X_all.values[test_index]
		y_train, y_test = y_all.values[train_index], y_all.values[test_index]
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		accuracy = accuracy_score(y_test, predictions)
		outcomes.append(accuracy)
		print("Fold {0} accuracy: {1}".format(fold, accuracy))
	mean_outcome = np.mean(outcomes)
	print("Mean Accuracy: {0}".format(mean_outcome))

run_kfold(clf)

#-- 8. Predict the Actual Test Data

ids = data_test['PassengerId']
predictions = clf.predict(data_test.drop('PassengerId', axis = 1))

output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})

print(output.head())





