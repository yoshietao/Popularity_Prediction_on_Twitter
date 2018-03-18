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
	print ("RMSE: ", (rmse_test/10)**0.5, "MAE: ", mae_test/10)

def predict_period(x,y):
	lrm = LinearRegression(fit_intercept=True, normalize=False)
	k_fold_rmse(lrm,x,y)
	
	'''
	lrm = lrm.fit(x, y)
	y_pred = lrm.predict(x)
	df = pd.DataFrame({'Actual': sum(y), 'Predict': sum(y_pred)})
	print (df)
	'''

	svm = SVC()
	k_fold_rmse(svm,x,y)
	rf = RandomForestClassifier()
	k_fold_rmse(rf,x,y)

def q1_4(filename):
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
	q1_4('tweets_#gohawks')
	#q1_4('tweets_#gopatriots')
	#q1_4('tweets_#nfl')
	#q1_4('tweets_#patriots')
	#q1_4('tweets_#sb49')
	#q1_4('tweets_#superbowl')
	#q1_4_2()

def q1_5_1(filename):
	x, y = func.load_q1_5(filename)

	'''
	print ("For ", filename)
	dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, dx6, dy6 = func.load_q1_5(filename)
	print ("First hour: ")
	predict_period(dx1, dy1)
	print ("Second hour: ")
	predict_period(dx2, dy2)
	print ("Third hour: ")
	predict_period(dx3, dy3)
	print ("Fourth hour: ")
	predict_period(dx4, dy4)
	print ("Fifth hour: ")
	predict_period(dx5, dy5)
	print ("Sixth hour: ")
	predict_period(dx6, dy6)
`	'''
def q1_5():

	q1_5_1('sample1_period1')
	q1_5_1('sample2_period2')
	q1_5_1('sample3_period3')
	q1_5_1('sample4_period1')
	q1_5_1('sample5_period1')
	q1_5_1('sample6_period2')
	q1_5_1('sample7_period3')
	q1_5_1('sample8_period1')
	q1_5_1('sample9_period2')
	q1_5_1('sample10_period3')

	#dp1x, dp1y, dp2x, dp2y, dp3x, dp3y = func.load_q1_5_stack()

	'''
	print ("Period 1:")
	predict_period(dp1x,dp1y)
	print ("Period 2:")
	predict_period(dp2x,dp2y)
	print ("Period 3:")
	predict_period(dp3x,dp3y)
	'''

def Q1_5():
	q1_5()


def main():
	#Q1_1()
	#Q1_2()
	Q1_3()
	#Q1_4()
	Q1_5()


if __name__ == '__main__':
	main()


