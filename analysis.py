'''
Rushikesh Joshi and KC Emezie
Professor Yue
CS155

Kaggle Mini-Project: Sentiment Analysis
'''

import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

# Load the entire data set as input
inpFile = open("data/training_data.txt", "r")

# Extract the bag of words which is the first line as a list
words = inpFile.readline().rstrip().split("|")

# Extract the rest of the data so that we can parse it
training = np.genfromtxt("data/training_data.txt", delimiter="|", skip_header=1)
test = np.genfromtxt("data/testing_data.txt", delimiter="|", skip_header=1)

X = training[:, :1000]
Y = training[:, 1000]

N = len(training)

inv_doc_freq = np.zeros(1000)
for i in range(len(inv_doc_freq)):
	total = sum(X[:, i])
	if total == 0:
		inv_doc_freq[i] = 0
	else:
		inv_doc_freq[i] = math.log(N / sum(X[:, i]))

# Data normalization
for i in range(len(X)):
	max_freq = max(X[i])
	if max_freq == 0:
		term_freq = X[i]
	else:
		term_freq = 0.5 + 0.5*(X[i] / max_freq)
	X[i] = term_freq * inv_doc_freq

# Testing with Decision Tree Classifier
'''
# Create leaf sizes to iterate over
leaf = []
for i in range(1, 26):
	leaf.append(i)

# Create tree depth to iterate over
depths = []
for i in range(2, 21):
	depths.append(i)

leaf_errors = []
depth_errors = []

# Default is Gini in DTC classifier
for size in leaf:
	tree = dtc(min_samples_leaf=size)
	tree.fit(X, Y)
	scores = cross_val_score(tree, X, Y)
	leaf_errors.append((scores.mean(), size))

print "Max leaf score: ", max(leaf_errors)
'''

# Default is Gini in DTC classifier
'''
for depth in depths:
	tree = dtc(max_depth=depth)
	tree.fit(X, Y)
	scores = cross_val_score(tree, X, Y)
	depth_errors.append((scores.mean(), size))

print "Max depth score: ", max(depth_errors)
'''

# Generate results for Decision Tree Classifier
'''
tree = dtc(min_samples_leaf=2)

tree.fit(X, Y)
vals = tree.predict(test)
'''

# SVM trial
'''
svr = SVC(C=0.1, kernel='linear', coef0=1)
model = svr.fit(X, Y)
scores = cross_val_score(model, X, Y)
print scores.mean()
'''

# Print out data
'''
vals = model.predict(test)

output = open("submissions/results_svm_1.txt", "w")
output.write("Id,Prediction" + "\n")
for i in range(1, len(vals) + 1):
	output.write(str(i) + "," + str(int(vals[i - 1])) + "\n")

inpFile.close()
output.close()
'''






