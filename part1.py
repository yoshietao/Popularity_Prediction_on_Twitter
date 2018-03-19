import func
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

def Q1_1():
	d_nfl = func.load_q1_1('tweets_#nfl')
	d_superbowl = func.load_q1_1('tweets_#superbowl')
	print(d_nfl)
	q1_1(d_nfl)
	q1_1(d_superbowl)

def q1_1(d):
	d1 = OrderedDict()
	for tweet in d:
		d1[tweet[0]] = d1[tweet[0]]+1 if tweet[0] in d1 else 1
	print('Average number of tweets per hour:', sum(d1.values())/len(d1))
	print('Average number of followers of users posting the tweets:', sum(d[:,1].astype('float'))/d.shape[0])
	print('Average number of retweets:',sum(d[:,2].astype('int'))/d.shape[0])

def Q1_2():
	d_nfl = func.load_q1_2('tweets_#nfl')
	d_superbowl = func.load_q1_2('tweets_#superbowl')
	func.plot_histogram(d_nfl[1])
	func.plot_histogram(d_superbowl[1])
	q2(d_nfl)
	q2(d_superbowl)

def q2(d,select=None):		#d[0] = x, d[1] = y
	lrm = LinearRegression(fit_intercept=True, normalize=False)
	#x,y = shuffle(d[0],d[1],random_state=42)
	x,y = d[0],d[1]
	results = sm.OLS(y, x).fit()
	
	print(results.summary())
	y_pred = lrm.fit(x,y).predict(x)
	print(mean_squared_error(y,lrm.fit(x,y).predict(x))**0.5)
	#func.analysis_q2(y,y_pred)

	fig,ax = plt.subplots()
	ax.scatter(np.arange(len(y)),y,s=5,label='true')
	ax.scatter(np.arange(len(y)),y_pred,s=5,label='fitted')
	ax.legend(loc=2)
	plt.show()

	if select is not None:
		for i in select:
			fig,ax = plt.subplots()
			ax.scatter(x[:,i],y,s=4,label='x'+str(i+1))
			ax.legend(loc=2)
			plt.show()

def k_fold_rmse(model,x,y):
	kf = KFold(n_splits=10, random_state=42)
	mae_test = 0
	rmse_test  = 0
	for train_index, test_index in kf.split(x):
		X_train, X_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		model = model.fit(X_train,y_train)
		y_test_pred  = model.predict(X_test)
		rmse_test  += mean_squared_error(y_test,y_test_pred)
		mae_test   += mean_absolute_error(y_test,y_test_pred)
<<<<<<< HEAD
	print ("RMSE: ", (rmse_test/10)**0.5, "MAE: ", mae_test/10)
=======
	print('RMSE is ', (rmse_test/10)**0.5)
	print('MAE is ', mae_test/10)
>>>>>>> d6baec6f77aec9cf0a532458ff4d5a15f278c909

def predict_period(x,y):
	lrm = LinearRegression(fit_intercept=True, normalize=False)
	k_fold_rmse(lrm,x,y)
	svm = SVC()
	k_fold_rmse(svm,x,y)
	rf = RandomForestClassifier()
	k_fold_rmse(rf,x,y)

def q1_4(filename):
	print('Q1_4 is predicting ' + filename + ' ...')
	d1x, d1y, d2x, d2y, d3x, d3y = func.load_q1_4(filename)
	predict_period(d1x,d1y)
	predict_period(d2x,d2y)
	predict_period(d3x,d3y)

def q1_4_2():
	d1x, d1y, d2x, d2y, d3x, d3y = func.load_q1_4_2()
	print(d1x.shape,d1y.shape)
	lrm = LinearRegression(fit_intercept=True, normalize=False)
	rf = RandomForestClassifier()
	k_fold_rmse(rf,d1x,d1y)
	k_fold_rmse(rf,d2x,d2y)
	k_fold_rmse(lrm,d3x,d3y)

def Q1_3():
	d_nfl = func.load_q1_3('tweets_#nfl')
	d_superbowl = func.load_q1_3('tweets_#superbowl')
	q2([d_nfl[0],d_nfl[1]],[0,3,4])
	q2([d_superbowl[0],d_superbowl[1]],[0,3,4])

def Q1_4():
	#q1_4('tweets_#gopatriots')
	#q1_4('tweets_#nfl')
	#q1_4('tweets_#patriots')
	#q1_4('tweets_#sb49')
	#q1_4('tweets_#superbowl')
	#q1_4_2()
	#q1_4_2()

def Q3():
	hashtags = ['tweets_#gohawks', 'tweets_#gopatriots', 'tweets_#nfl', 'tweets_#patriots', 'tweets_#sb49', 'tweets_#superbowl']

	for objective in hashtags:
		empty = True
		augmented_x1, augmented_x2, augmented_x3 = [], [], []
		y1, y2, y3 = func.load_q1_4(objective)[1], func.load_q1_4(objective)[3], func.load_q1_4(objective)[5]
		for other in hashtags:
			if other != objective:
				x1, x2, x3 = func.load_q1_4(other)[0], func.load_q1_4(other)[2], func.load_q1_4(other)[4]
				if empty==True:
					augmented_x1, augmented_x2, augmented_x3 = x1, x2, x3
					empty = False
				else:
					augmented_x1, augmented_x2, augmented_x3 = np.hstack((augmented_x1, x1)), np.hstack((augmented_x2, x2)), np.hstack((augmented_x3, x3))
		print('Q3 is predicting ' + objective + ' ...')
		predict_period(augmented_x1, y1)
		predict_period(augmented_x2, y2)
		predict_period(augmented_x3, y3)

def q1_5_1(filename, period):
	x, y = func.load_q1_5(filename, period)
	print ("=====================")
	print (filename)
	print ("=====================")
	print ("--------Linear Regression-------")
	for i in range(0, len(x)):
		x_train, y_train = [], []
		for j in range(0, len(x)):
			if i != j:
				x_train.append(x[j])
				y_train.append(y[j])
		model = LinearRegression(fit_intercept=True, normalize=False)		
		model = model.fit(x_train, y_train)
		x_test = [x[i]]
		y_pred = model.predict(x_test)
		print (i, y_pred, x[i][0], y_pred-x[i][0])

	print ("--------SVM--------")
	for i in range(0, len(x)):
		x_train, y_train = [], []
		for j in range(0, len(x)):
			if i != j:
				x_train.append(x[j])
				y_train.append(y[j])
		model = SVC()	
		model = model.fit(x_train, y_train)
		x_test = [x[i]]
		y_pred = model.predict(x_test)
		print (i, y_pred, x[i][0], y_pred-x[i][0])

	print ("--------Random Forest--------")
	for i in range(0, len(x)):
		x_train, y_train = [], []
		for j in range(0, len(x)):
			if i != j:
				x_train.append(x[j])
				y_train.append(y[j])
		model = RandomForestClassifier()
		model = model.fit(x_train, y_train)
		x_test = [x[i]]
		y_pred = model.predict(x_test)
		print (i, y_pred, x[i][0], y_pred-x[i][0])

def Q1_5():

	q1_5_1('sample1_period1', 1)
	q1_5_1('sample2_period2', 2)
	q1_5_1('sample3_period3', 3)
	q1_5_1('sample4_period1', 1)
	q1_5_1('sample5_period1', 1)
	q1_5_1('sample6_period2', 2)
	q1_5_1('sample7_period3', 3)
	q1_5_1('sample8_period1', 1)
	q1_5_1('sample9_period2', 2)
	q1_5_1('sample10_period3', 3)


def main():
	#Q1_1()
	#Q1_2()
	#Q1_3()
	#Q1_4()
	Q1_5()

	Q3()



if __name__ == '__main__':
	main()


