'''
	Linear Regression for stock price predictions, using sklearn
	@author: Pranjal Verma
'''

import numpy as np
import quandl, math, pickle
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression

# to ignore that annoying warning
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# train LR classifier
def trainLR():

	# getting stock data for Google
	df = quandl.get('WIKI/GOOGL')
	df.fillna(-99999, inplace=True)

	# creating and filtering useful features
	df['HL_%'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
	df['%_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
	df = df[['Adj. Close', 'HL_%', '%_change', 'Adj. Volume']]

	# feature to be predicted and days to be shifted
	forecast_col = 'Adj. Close'
	forecast_shift = int(math.ceil(0.01*len(df)))

	# creating label for features and dropping NaN label rows that occur cuz of shift
	df['label'] = df[forecast_col].shift(-forecast_shift)
	df.dropna(inplace=True)

	# features, labels, and normalisation
	X, y = np.array(df.drop(columns=['label'])), np.array(df['label'])
	X = preprocessing.scale(X)

	# seperating out test set
	# test set should be taken from the end, at least in this case
	# think of how stocks work
	X_test, y_test = X[-forecast_shift:], y[-forecast_shift:]
	X, y = X[:-forecast_shift], y[:-forecast_shift]

	# splitting data; training and CV sets
	X_train, X_cv, y_train, y_cv = model_selection.train_test_split(X, y, test_size=0.2)

	# creating classifier and fitting data
	clf = LinearRegression(n_jobs=-1)
	clf.fit(X_train, y_train)

	# saving trained classifier
	with open('LinearRegression.pickle', 'wb') as fp:
		pickle.dump(clf, fp)

	# getting accuracy of classifier
	accuracy = clf.score(X_cv, y_cv)
	print('Accuracy: ' + str(accuracy))

	return X_test, y_test

# get predictions
def predictLR(X_test, y_test):

	# loading trained LR classifier
	LR_model = open('LinearRegression.pickle', 'rb')
	clf = pickle.load(LR_model)

	# getting forecasts
	forecast = clf.predict(X_test)
	print('Forecast          Truth')
	for i in range(len(X_test)):
		print(forecast[i], y_test[i])

# init
if __name__ == '__main__':
	X_test, y_test = trainLR()
	predictLR(X_test, y_test)
