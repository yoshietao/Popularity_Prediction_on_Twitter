import func
from collections import OrderedDict
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt

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

def Q1_3():
	d_nfl = func.load_q1_3('tweets_#nfl')
	d_superbowl = func.load_q1_3('tweets_#superbowl')
	q2([d_nfl[0],d_nfl[1]],[0,2,3])
	q2([d_superbowl[0],d_superbowl[1]],[0,2,3])

def main():
	#Q1_1()
	#Q1_2()
	Q1_3()


if __name__ == '__main__':
	main()


