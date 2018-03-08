import datetime, time
import pytz
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os 
pst_tz = pytz.timezone('US/Pacific')

'''
class Data:
	def __init__(self):
		with open('tweets_#nfl.txt') as json_data:
			self.nfl = [json.loads(line) for line in json_data]
		with open('tweets_#superbowl.txt') as json_data:
			self.superbowl = [json.loads(line) for line in json_data]
'''
def load_q1_1(filename):
	if Path('./data/'+filename+'Q1_1.npy').exists():
		print('Already have Q1_1 data.')
		return np.load('./data/'+filename+'Q1_1.npy')
	else:
		print('Parsing Q1_1 data...')
		d = []
		with open(filename+'.txt') as data:
			for line in data:
				line = json.loads(line)
				d.append([datetime.datetime.fromtimestamp(line['citation_date'], pst_tz).strftime('%Y-%m-%d %H:%M:%S')[5:13], line['author']['followers'], line['metrics']['citations']['total']])
		np.save('./data/'+filename+'Q1_1.npy',np.array(d))
		return np.load('./data/'+filename+'Q1_1.npy')

def load_q1_2(filename):
	if Path('./data/'+filename+'Q1_2x.npy').exists():
		print('Already have Q1_2 data.')
		return np.load('./data/'+filename+'Q1_2x.npy'),np.load('./data/'+filename+'Q1_2y.npy')
	else:
		print('Parsing Q1_2 data...')
		os.makedirs('./data')
		x = OrderedDict()
		y = OrderedDict()
		with open(filename+'.txt') as data:
			for line in data:
				line = json.loads(line)
				index = datetime.datetime.fromtimestamp(line['citation_date'], pst_tz).strftime('%Y-%m-%d %H:%M:%S')[5:13]
				if index not in x:
					x[index] = [0,0,0,0,int(index[6:8])]
				x[index][0] += 1
				x[index][1] += int(line['metrics']['citations']['total'])
				x[index][2] += float(line['author']['followers'])
				if float(line['author']['followers']) > x[index][3]:
					x[index][3] = float(line['author']['followers'])
		x = np.array(list(x.values())).astype('int')
		y = x[:,0]
		np.save('./data/'+filename+'Q1_2x.npy',x[:-1,:])
		np.save('./data/'+filename+'Q1_2y.npy',y[1:])
		print(x,y,x.shape,y.shape)
		return np.load('./data/'+filename+'Q1_2x.npy'), np.load('./data/'+filename+'Q1_2y.npy')

def plot_histogram(d):
	fig,ax = plt.subplots()
	plt.bar(np.arange(len(d)),list(d.values()))
	ax.set_xlabel('hours')
	ax.set_ylabel('tweets')
	ax.set_title('Number of tweets in hour')
	plt.show()

def main():
	d_nfl = load_d('tweets_#nfl')
	d_superbowl = load_d('tweets_#superbowl')

if __name__ == '__main__':
	main()
	