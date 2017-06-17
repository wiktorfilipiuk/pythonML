import pandas as pd
from sklearn import preprocessing

def simplify_ages(df):
	df.Age = df.Age.fillna(-0.5)
	bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
	group_names= ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
	categories = pd.cut(df.Age, bins, labels = group_names)
	df.Age = categories
	return(df)

def simplify_cabins(df):
	df.Cabin = df.Cabin.fillna('N')
	df.Cabin = df.Cabin.apply(lambda x: x[0])
	return(df)

def simplify_fares(df):
	df.Fare = df.Fare.fillna(-0.5)
	bins = (-1, 0, 8, 15, 31, 1000)
	group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
	categories = pd.cut(df.Fare, bins, labels = group_names)
	df.Fare = categories
	return(df)

def format_name(df):
	df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
	df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
	#df['NamePrefix'] = df.Name.apply(lambda x: [item for item in x.split(' ') if "." in item][0])
	return(df)

def drop_features(df):
	return(df.drop(['Ticket', 'Name', 'Embarked'], axis = 1))


def encode_features(df_train, df_test):
	features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
	df_combined = pd.concat([df_train[features], df_test[features]])
	
	for feature in features:
		le = preprocessing.LabelEncoder()
		le = le.fit(df_combined[feature])
		df_train[feature] = le.transform(df_train[feature])
		df_test[feature] = le.transform(df_test[feature])
	
	return df_train, df_test

#












