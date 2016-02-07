'''
Rushikesh Joshi and KC Emezie
Professor Yue
CS155

Kaggle Mini-Project: Sentiment Analysis
'''

import numpy as np
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

# Load the entire data set as input
inpFile = open("data/training_data.txt", "r")

# Extract the bag of words which is the first line as a list
words = inpFile.readline().rstrip().split("|")

# Extract the rest of the data so that we can parse it
training = np.genfromtxt("data/training_data.txt", delimiter="|", skip_header=1)
test = np.genfromtxt("data/testing_data.txt", delimiter="|", skip_header=1)

X = training[:, :1000]
Y = training[:, 1000]

training_data = []

'''
for i in len(data):
	counts = data[i]
	tmp_lst = []
	label = 0
	for j in range(len(counts) - 1):
		tmp_list += counts[j] * [words[j]]
	training_data.append()
'''


# Create leaf sizes to iterate over
leaf = []
for i in range(1, 26):
	leaf.append(i)

# Create tree depth to iterate over
depths = []
for i in range(2, 21):
	depths.append(i)

'''
tree = dtc(min_samples_leaf=10)

tree.fit(X, Y)
vals = tree.predict(test)
'''

leaf_errors = []
depth_errors = []

# Default is Gini in DTC classifier
for size in leaf:
	tree = dtc(min_samples_leaf=size)
	tree.fit(X, Y)
	scores = cross_val_score(tree, X, Y)
	leaf_errors.append((scores.mean(), size))
	print scores.mean()

print "Max lead score: ", max(leaf_errors)

# Default is Gini in DTC classifier
for depth in depths:
	tree = dtc(max_depth=depth)
	tree.fit(X, Y)
	scores = cross_val_score(tree, X, Y)
	depth_errors.append((scores.mean(), size))

print "Max depth score: ", max(depth_errors)


'''
output = open("submissions/results.txt", "w")
output.write("Id,Prediction" + "\n")
for i in range(1, len(vals) + 1):
	output.write(str(i) + "," + str(int(vals[i - 1])) + "\n")
'''

inpFile.close()
#output.close()







