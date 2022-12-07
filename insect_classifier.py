from skmultiflow.trees import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier
from skmultiflow.lazy import SAMKNNClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier, AdditiveExpertEnsembleClassifier
from skmultiflow.drift_detection import PageHinkley
import skmultiflow as flow
from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn import preprocessing
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle



def arffToCSV(filepath, new_path):
	data = arff.loadarff(filepath)
	train= pd.DataFrame(data[0])
	le = preprocessing.LabelEncoder()
	train['class'] = le.fit_transform(train['class'])
	catCols = [col for col in train.columns if train[col].dtype=="O"]
	train[catCols]=train[catCols].apply(lambda x: x.str.decode('utf8'))
	train.to_csv(new_path,index=False)



# arffToCSV('./data/INSECTS-abrupt_balanced_norm.arff', './data/abrupt_balanced_norm.csv')
# arffToCSV('./data/INSECTS-gradual_balanced_norm.arff', './data/gradual_balanced_norm.csv')
# arffToCSV('./data/INSECTS-incremental_balanced_norm.arff', './data/incremental_balanced_norm.csv')

abrupt_stream = ("abrupt", flow.data.FileStream('./data/abrupt_balanced_norm.csv', target_idx=-1, n_targets=1))
gradual_stream = ("gradual", flow.data.FileStream('./data/gradual_balanced_norm.csv', target_idx=-1, n_targets=1))
incremental_stream = ("incremental", flow.data.FileStream('./data/incremental_balanced_norm.csv', target_idx=-1, n_targets=1))

streams = [abrupt_stream, gradual_stream, incremental_stream]

def sliding_window_majority(stream, label, dd=PageHinkley, showGraph=False):

	stream.reset()
	ddm = dd()
	total_pred=0
	window = []
	score_window = []
	total_preq_acc = []
	time_of_pred = []
	time_of_change = []

	for i in range(1000):
		new_sample = stream.next_sample(1)[1]
		window.append(new_sample[0])
		ddm.add_element(new_sample)

	while(stream.has_more_samples()):

		new_sample = stream.next_sample(1)

		ddm.add_element(new_sample[1])
		if ddm.detected_change():			
			time_of_change.append(total_pred)

		pred = Counter(window).most_common(1)[0][0]

		if(pred==new_sample[1][0]):
			pred_score = 1
		else:
			pred_score = 0

		if(len(window)==1000):
			window = window[1:1000]

		window.append(new_sample[1][0])
		
		if(len(score_window)==1000):
			score_window = score_window[1:1000]

		score_window.append(pred_score)

		preq_acc = sum(score_window)/len(score_window)
		total_preq_acc.append(preq_acc)
		
		total_pred+=1
		time_of_pred.append(total_pred)

	avg_preq_acc = sum(total_preq_acc)/len(total_preq_acc)
	print(avg_preq_acc)
	if(showGraph):
		for i in range(len(time_of_change)):
			plt.axvline(x = time_of_change[i], color = 'r')
		plt.plot(time_of_pred, total_preq_acc, label=label)
		plt.legend(loc="upper left")

	return (label, avg_preq_acc)

def sliding_window_no_change(stream, label, dd=PageHinkley, showGraph=False):
	stream.reset()
	ddm = dd()
	total_pred=0
	window = []
	total_preq_acc = []
	time_of_pred = []
	time_of_change = []

	for i in range(1000):
		new_sample = stream.next_sample(1)[1]
		ddm.add_element(new_sample)
		previous_result = new_sample

	while(stream.has_more_samples()):

		new_sample = stream.next_sample(1)

		ddm.add_element(new_sample[1])
		if ddm.detected_change():
			time_of_change.append(total_pred)
	
		pred =  previous_result

		if(pred==new_sample[1][0]):
			pred_score = 1
		else:
			pred_score = 0

		if(len(window)==1000):
			window = window[1:1000]

		window.append(pred_score)

		preq_acc = sum(window)/len(window)

		total_preq_acc.append(preq_acc)
		
		previous_result = new_sample[1]
		
		total_pred+=1
		time_of_pred.append(total_pred)

	avg_preq_acc = sum(total_preq_acc)/len(total_preq_acc)
	print(avg_preq_acc)
	if(showGraph):
		for i in range(len(time_of_change)):
			plt.axvline(x = time_of_change[i], color = 'r')
		plt.plot(time_of_pred, total_preq_acc, label=label)
		plt.legend(loc="upper left")

	return (label, avg_preq_acc)	

def sliding_window(stream, label, clf, dd=PageHinkley, showGraph=False, drifDetect=False, **kwargs):
	stream.reset()

	ddm = dd()
	model = clf(**kwargs)
	total_pred=0
	time_of_pred = []
	window = []
	total_preq_acc = []
	time_of_change = []

	new_sample = stream.next_sample(1000)
	window_warmup = new_sample[0]
	labels_warmup = new_sample[1]

	model = model.partial_fit(window_warmup, labels_warmup)

	for i in labels_warmup:
		ddm.add_element(i)

	while(stream.has_more_samples()):
	
		new_sample = stream.next_sample(1)
		pred = model.predict(new_sample[0])

		ddm.add_element(new_sample[1])
		if ddm.detected_change():
			if(drifDetect):
				model = clf(**kwargs)
			time_of_change.append(total_pred)

		if(pred[0]==new_sample[1][0]):
			pred_score = 1
		else:
			pred_score = 0

		if(len(window)==1000):
			window.pop(0)

		window.append(pred_score)

		preq_acc = sum(window)/len(window)
		
		total_preq_acc.append(preq_acc)
		
		model = model.partial_fit(new_sample[0], new_sample[1])
		total_pred+=1
		time_of_pred.append(total_pred)
	
	avg_preq_acc = sum(total_preq_acc)/len(total_preq_acc)
	print(avg_preq_acc)

	if(showGraph):
		for i in range(len(time_of_change)):
			plt.axvline(x = time_of_change[i], color = 'r')
		plt.plot(time_of_pred, total_preq_acc, label=label)
		plt.legend(loc="upper left")

	return (label, avg_preq_acc)


def createTable():

	with open('./pickles/scores_q1_3.pickle', 'rb') as handle:
		scores_q1_3 = pickle.load(handle)
	with open('./pickles/scores_q4_5.pickle', 'rb') as handle:
		scores_q4_5 = pickle.load(handle)

	fig, ax = plt.subplots()
	fig.patch.set_visible(False)
	ax.axis('off')
	ax.axis('tight')
	df = pd.DataFrame(columns=['stream'] + [i[0] for i in scores_q1_3['abrupt']] + [scores_q4_5['abrupt'][1][0]])

	x=0
	for stream in streams:
		df.loc[x] = [stream[0]] + [i[1] for i in scores_q1_3[stream[0]] + [scores_q4_5[stream[0]][1]]]
		x+=1

	table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')	
	table.auto_set_font_size(False) 
	table.set_fontsize(11)
	fig.tight_layout()
	plt.show()

### QUESTION 1 - 3
def q1_3():
	
	dic_scores = {}
	for stream in streams:

		score_list = []

		print("\nprocessing stream:", stream[0])
		plt.figure()

		print("\nNo Change")
		score_list.append(sliding_window_no_change(stream[1], "No Change", showGraph=True))


		print("\nMajority")
		score_list.append(sliding_window_majority(stream[1], "Majority", showGraph=True))


		print("\nHoeffdingTreeClassifier")
		score_list.append(sliding_window(stream[1], "HoeffdingTree", HoeffdingTreeClassifier, showGraph=True))


		print("\nSAMKNNClassifier")
		score_list.append(sliding_window(stream[1], "SAMKNN", SAMKNNClassifier, showGraph=True))


		print("\nHoeffdingAdaptiveTreeClassifier")
		score_list.append(sliding_window(stream[1], "HoeffdingAdaptiveTree", HoeffdingAdaptiveTreeClassifier, showGraph=True))


		print("\nAdaptiveRandomForestClassifier")
		score_list.append(sliding_window(stream[1], "AdaptiveRandomForest", AdaptiveRandomForestClassifier, showGraph=True))

		print("\nAdditiveExpertEnsembleClassifier")
		score_list.append(sliding_window(stream[1], "AdditiveExpertEnsemble", AdditiveExpertEnsembleClassifier, showGraph=True))
		
		handles, labels = plt.gca().get_legend_handles_labels()
		red_patch = mpatches.Patch(color='red', label='change detected')
		handles.append(red_patch)
		plt.legend(handles=handles)
		plt.title(stream[0])
		plt.xlabel("instance number")
		plt.ylabel("prequential accuracy")

		dic_scores[stream[0]] = score_list

	with open('./pickles/scores_q1_3.pickle', 'wb') as handle:
		pickle.dump(dic_scores, handle)
	
	#plt.show()


# QUESTION 4, 5
def q4_5():

	dic_scores = {}

	for stream in streams:

		score_list = []

		print("\nprocessing stream:", stream[0])
		plt.figure() 

		print("\nHoeffdingTreeClassifier")
		score_list.append(sliding_window(stream[1], "HoeffdingTree \nNo Drift Detect", HoeffdingTreeClassifier, showGraph=True))

		print("\nHoeffdingTreeClassifier with Drift Detect")

		score_list.append(sliding_window(stream[1], "HoeffdingTree \nWith Drift Detect", HoeffdingTreeClassifier, showGraph=True, drifDetect=True))

		handles, labels = plt.gca().get_legend_handles_labels()
		red_patch = mpatches.Patch(color='red', label='change detected')
		handles.append(red_patch)
		plt.legend(handles=handles)
		plt.title(stream[0])
		plt.xlabel("instance number")
		plt.ylabel("prequential accuracy")

		dic_scores[stream[0]] = score_list

	with open('./pickles/scores_q4_5.pickle', 'wb') as handle:
		pickle.dump(dic_scores, handle)
			
	#plt.show()

q1_3()
# q4_5()
createTable()
