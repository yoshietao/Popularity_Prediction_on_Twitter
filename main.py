import func
from collections import OrderedDict
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle




def Q1(d):
	d1 = OrderedDict()
	for tweet in d:
		d1[tweet[0]] = d1[tweet[0]]+1 if tweet[0] in d1 else 1
	print('Average number of tweets per hour:', sum(d1.values())/len(d1))
	print('Average number of followers of users posting the tweets:', sum(d[:,1].astype('float'))/d.shape[0])
	print('Average number of retweets:',sum(d[:,2].astype('int'))/d.shape[0])
	func.plot_histogram(d1)

def Q2():
	d_nfl = func.load_q1_2('tweets_#nfl')
	d_superbowl = func.load_q1_2('tweets_#superbowl')
	q2(d_nfl)
	q2(d_superbowl)

def q2(d):		#d[0] = x, d[1] = y
	lrm = LinearRegression(fit_intercept=True, normalize=True)
	x,y = shuffle(d[0],d[1],random_state=42)







def main():
	#d_nfl = func.load_q1_1('tweets_#nfl')
	#d_superbowl = func.load_q1_1('tweets_#superbowl')
	#Q1(d_nfl)
	#Q1(d_superbowl)
	Q2()


if __name__ == '__main__':
	main()