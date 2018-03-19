import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, auc, roc_curve, accuracy_score
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from string import punctuation
import collections
import itertools
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime, time
import pytz
import math 
pst_tz = pytz.timezone('US/Pacific')


def load_q3(filename):

	if Path('./data/'+filename+'Q3_x1.npy').exists():
		print('Already have Q3 data.')
		return np.load('./data/'+filename+'Q3_x1.npy'),np.load('./data/'+filename+'Q3_y1.npy'),np.load('./data/'+filename+'Q3_x3.npy'),np.load('./data/'+filename+'Q3_y3.npy')
	else:
		print('Parsing Q3 data: '+filename+'...')
		if not Path('./data').exists():
			print('Making directory: data/')
			os.makedirs('./data')
		
		x1, y1, x3, y3 = [], [], [], []
		sid = SentimentIntensityAnalyzer()

		with open('../tweet_data/'+filename+'.txt') as data:
			for line in data:
				line = json.loads(line)
				index = datetime.datetime.fromtimestamp(line['citation_date'], pst_tz).strftime('%Y-%m-%d %H:%M:%S')[5:13]
				index1 = int(index[:2])*31*24+int(index[3:5])*24+int(index[6:8])
				text = line['title']
				location = line['tweet']['user']['location']
				location_label = 0.0
				if location.find('Washington') is not -1 or location.find('WA') is not -1 or location.find('Seattle') is not -1:
					location_label = 0.0
				elif location.find('Massachusetts') is not -1 or location.find('MA') is not -1 or location.find('Boston') is not -1:
					location_label = 1.0
				else: 
					continue

				sentiment = []
				ss = sid.polarity_scores(text) ## extracting sentiment vector
				for k in ss:
					sentiment.append(ss[k])

				if index1 < 1520:
					x1.append(sentiment)
					y1.append(location_label)
				elif index1 > 1532:
					x3.append(sentiment)
					y3.append(location_label)
				
		x1 = np.array(x1)
		y1 = np.array(y1)
		x3 = np.array(x3)
		y3 = np.array(y3)

		np.save('./data/'+filename+'Q3_x1.npy',x1)
		np.save('./data/'+filename+'Q3_y1.npy',y1)
		np.save('./data/'+filename+'Q3_x3.npy',x3)
		np.save('./data/'+filename+'Q3_y3.npy',y3)
		
		return np.load('./data/'+filename+'Q3_x1.npy'),np.load('./data/'+filename+'Q3_y1.npy'),np.load('./data/'+filename+'Q3_x3.npy'),np.load('./data/'+filename+'Q3_y3.npy')

def svm_classifier(x, y):

	svm_clf = svm.SVC(C=1000, kernel='linear', probability=True)
	svm_clf.fit(x, y)
	y_pred = svm_clf.predict(x)
	print(metrics.accuracy_score(y, y_pred))


def Q3_extra(filename):

	x1, y1, x3, y3 = load_q3(filename)

	print(x1, y1, x3, y3)
	print(np.shape(x1), np.shape(y1), np.shape(x3), np.shape(y3))

	print('accuracy in period 1 is:')
	svm_classifier(x1, y1)
	print('accuracy in period 3 is:')
	#svm_classifier(x3, y3)



def main():

	Q3_extra('tweets_#superbowl')



if __name__ == '__main__':
	main()
