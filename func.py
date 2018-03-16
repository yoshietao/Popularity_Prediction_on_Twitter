import datetime, time
import pytz
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import math 
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
		if not Path('./data').exists():
			print('Making directory: data/')
			os.makedirs('./data')
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
		if not Path('./data').exists():
			print('Making directory: data/')
			os.makedirs('./data')
		x = OrderedDict()
		y = OrderedDict()
		for i in range(14,32):
			for j in range(24):
				x['01-'+str(i)+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		for i in range(1,7):
			for j in range(24):
				x['02-'+'{0:02d}'.format(i)+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		for j in range(11):
			x['02-07'+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		with open(filename+'.txt') as data:
			for line in data:
				line = json.loads(line)
				index = datetime.datetime.fromtimestamp(line['citation_date'], pst_tz).strftime('%Y-%m-%d %H:%M:%S')[5:13]
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

def load_q1_3(filename):
	if Path('./data/'+filename+'Q1_3x.npy').exists():
		print('Already have Q1_3 data.')
		return np.load('./data/'+filename+'Q1_3x.npy'),np.load('./data/'+filename+'Q1_3y.npy')
	else:
		print('Parsing Q1_3 data...')
		if not Path('./data').exists():
			print('Making directory: data/')
			os.makedirs('./data')
		x = OrderedDict()
		y = OrderedDict()
		for i in range(14,32):
			for j in range(24):
				x['01-'+str(i)+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		for i in range(1,7):
			for j in range(24):
				x['02-'+'{0:02d}'.format(i)+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		for j in range(11):
			x['02-07'+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		with open(filename+'.txt') as data:
			for line in data:
				line = json.loads(line)
				index = datetime.datetime.fromtimestamp(line['citation_date'], pst_tz).strftime('%Y-%m-%d %H:%M:%S')[5:13]
				x[index][0] += 1
				x[index][1] += int(line['metrics']['citations']['total'])
				x[index][2] += float(line['author']['followers'])
				x[index][3] += int(line['metrics']['momentum'])
				if int(line['tweet']['user']['friends_count']) > x[index][4]:
					x[index][4] = int(line['tweet']['user']['friends_count'])
				#x[index][3] += int(line['metrics']['impressions'])
				#if float(line['metrics']['impressions'])>x[index][3]:
				#	x[index][3] = float(line['metrics']['impressions'])
				#if float(line['author']['followers']) > x[index][4]:
				#	x[index][4] = float(line['author']['followers'])
		x = np.array(list(x.values())).astype('int')
		y = x[:,0]
		np.save('./data/'+filename+'Q1_3x.npy',x[:-1,:])
		np.save('./data/'+filename+'Q1_3y.npy',y[1:])
		#print(x,y,x.shape,y.shape)
		return np.load('./data/'+filename+'Q1_3x.npy'), np.load('./data/'+filename+'Q1_3y.npy')

def load_q1_4(filename):
	if Path('./data/'+filename+'Q1_4x3.npy').exists():
		print('Already have Q1_4 data.')
		return np.load('./data/'+filename+'Q1_4x1.npy'),np.load('./data/'+filename+'Q1_4y1.npy'),np.load('./data/'+filename+'Q1_4x2.npy'),np.load('./data/'+filename+'Q1_4y2.npy'),np.load('./data/'+filename+'Q1_4x3.npy'),np.load('./data/'+filename+'Q1_4y3.npy')
	else:
		print('Parsing Q1_4 data:'+filename+'...')
		if not Path('./data').exists():
			print('Making directory: data/')
			os.makedirs('./data')
		x1 = OrderedDict()
		y1 = OrderedDict()
		x2 = OrderedDict()
		y2 = OrderedDict()
		x3 = OrderedDict()
		y3 = OrderedDict()
		for i in range(14,32):
			for j in range(24):
				x1['01-'+str(i)+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		for j in range(8):
			x1['02-01'+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		for j in range(8,21):
			x2['02-01'+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		for j in range(21,24):
			x3['02-01'+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		for i in range(2,7):
			for j in range(24):
				x3['02-'+'{0:02d}'.format(i)+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		for j in range(11):
			x3['02-07'+' '+'{0:02d}'.format(j)] = [0,0,0,0,j]
		with open(filename+'.txt') as data:
			for line in data:
				line = json.loads(line)
				index = datetime.datetime.fromtimestamp(line['citation_date'], pst_tz).strftime('%Y-%m-%d %H:%M:%S')[5:13]
				index1 = int(index[:2])*31*24+int(index[3:5])*24+int(index[6:8])
				if index1 < 1520:
					x1[index][0] += 1
					x1[index][1] += int(line['metrics']['citations']['total'])
					x1[index][2] += float(line['author']['followers'])
					x1[index][3] += int(line['metrics']['momentum'])
					if int(line['tweet']['user']['friends_count']) > x1[index][4]:
						x1[index][4] = int(line['tweet']['user']['friends_count'])
				elif index1 > 1532:
					x3[index][0] += 1
					x3[index][1] += int(line['metrics']['citations']['total'])
					x3[index][2] += float(line['author']['followers'])
					x3[index][3] += int(line['metrics']['momentum'])
					if int(line['tweet']['user']['friends_count']) > x3[index][4]:
						x3[index][4] = int(line['tweet']['user']['friends_count'])
				else:
					x2[index][0] += 1
					x2[index][1] += int(line['metrics']['citations']['total'])
					x2[index][2] += float(line['author']['followers'])
					x2[index][3] += int(line['metrics']['momentum'])
					if int(line['tweet']['user']['friends_count']) > x2[index][4]:
						x2[index][4] = int(line['tweet']['user']['friends_count'])
		x1 = np.array(list(x1.values())).astype('int')
		x2 = np.array(list(x2.values())).astype('int')
		x3 = np.array(list(x3.values())).astype('int')
		y1 = x1[:,0]
		y2 = x2[:,0]
		y3 = x3[:,0]
		np.save('./data/'+filename+'Q1_4x1.npy',x1[:-1,:])
		np.save('./data/'+filename+'Q1_4x2.npy',x2[:-1,:])
		np.save('./data/'+filename+'Q1_4x3.npy',x3[:-1,:])
		np.save('./data/'+filename+'Q1_4y1.npy',y1[1:])
		np.save('./data/'+filename+'Q1_4y2.npy',y2[1:])
		np.save('./data/'+filename+'Q1_4y3.npy',y3[1:])
		#print(x,y,x.shape,y.shape)
		return np.load('./data/'+filename+'Q1_4x1.npy'),np.load('./data/'+filename+'Q1_4y1.npy'),np.load('./data/'+filename+'Q1_4x2.npy'),np.load('./data/'+filename+'Q1_4y2.npy'),np.load('./data/'+filename+'Q1_4x3.npy'),np.load('./data/'+filename+'Q1_4y3.npy')

def test(filename):
	x = OrderedDict()
	for i in range(14,32):
		for j in range(24):
			x['01-'+str(i)+' '+'{0:02d}'.format(j)] = [0,0]
	for i in range(1,7):
		for j in range(24):
			x['02-'+'{0:02d}'.format(i)+' '+'{0:02d}'.format(j)] = [0,0]
	for j in range(11):
		x['02-07'+' '+'{0:02d}'.format(j)] = [0,0]
	
	with open(filename+'.txt') as data:
		for line in data:
			line = json.loads(line)
			#print(line['metrics']['impressions'])
			index = datetime.datetime.fromtimestamp(line['citation_date'], pst_tz).strftime('%Y-%m-%d %H:%M:%S')[5:13]
			try:
				x[index][0] += 1
				if int(line['tweet']['user']['friends_count']) > x[index][1]:
					x[index][1] = int(line['tweet']['user']['friends_count'])
				#int(line['metrics']['impressions'])				-------->garbage
				#print(line['metrics']['impressions'])
				#y.append(line['metrics']['impressions'])
				#y.append(line['tweet']['user']['favourites_count'])
				#y.append(line['metrics']['citations']['total'])
				#y.append(line['metrics']['momentum'])
			except:
				print('xx')
				pass
			#index = datetime.datetime.fromtimestamp(line['citation_date'], pst_tz).strftime('%Y-%m-%d %H:%M:%S')[5:13]
			#x[index] = 1
	#print(x.keys())
	#fig,ax = plt.subplots()
	#ax.plot(np.arange(len(y)),y)
	#plt.show()
	x = np.array(list(x.values())).astype('int')
	#print(x)
	#y = [i/j for i,j in zip(x[:,1],x[:,0])]
	#for ind,i in enumerate(y):
	#	y[ind] = 0 if i=='nan' else i
	fig,ax = plt.subplots()
	ax.scatter(np.arange(x.shape[0]),x[:,0],s=5,label='y')
	ax.scatter(np.arange(x.shape[0]),x[:,1],s=5,label='feature')
	ax.legend(loc=2)
	plt.show()

def plot_histogram(d):
	fig,ax = plt.subplots()
	plt.bar(np.arange(len(d)),d)
	ax.set_xlabel('hours')
	ax.set_ylabel('tweets')
	ax.set_title('Number of tweets in hour')
	plt.show()

def analysis_q2(y_true,y_pred):
	for i,j in zip(y_true,y_pred):
		print(i,j)


def main():
	#load_q1_2('tweets_#nfl')
	#load_q1_2('tweets_#superbowl')
	test('tweets_#nfl')
	#test('tweets_#superbowl')

if __name__ == '__main__':
	main()
	