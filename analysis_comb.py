'''
Rushikesh Joshi and KC Emezie
Professor Yue
CS155

Kaggle Mini-Project: Sentiment Analysis
'''

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import VotingClassifier as VTC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score

# Create formatted submission file
def create_submission(filename, vals):
    output = open("submissions/" + filename, "w")
    output.write("Id,Prediction" + "\n")
    for i in range(1, len(vals) + 1):
	output.write(str(i) + "," + str(int(vals[i - 1])) + "\n")

    output.close()

# Load the entire data set as input
inpFile = open("data/training_data.txt", "r")

# Extract the bag of words which is the first line as a list
words = inpFile.readline().rstrip().split("|")

# Extract the rest of the data so that we can parse it
training_data = np.genfromtxt("data/training_data.txt", delimiter="|", skip_header=1)
test_data = np.genfromtxt("data/testing_data.txt", delimiter="|", skip_header=1)

X = training_data[:, :1000]
Y = training_data[:, 1000]

# Various Classifiers
dtc_min_samples_leaf = DTC(min_samples_leaf=15)
etc = ETC()
gbc = GBC()
rfc = RFC()
dtc_max_depth = DTC(max_depth=8)
nb  = BernoulliNB()
svc = SVC()

# Split Training Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3333)

# Compare individual classifiers
dtc_min_samples_leaf.fit(X_train, Y_train)
print "dtc_min_samples_leaf"
prediction1_train = dtc_min_samples_leaf.predict(X_train)
prediction1 = dtc_min_samples_leaf.predict(X_test)
print accuracy_score(prediction1, Y_test)

etc.fit(X_train, Y_train)
print "Extra Trees Classifier"
prediction2_train = etc.predict(X_train)
prediction2 = etc.predict(X_test)
print accuracy_score(prediction2, Y_test)

gbc.fit(X_train, Y_train)
print "GBC"
prediction3_train = gbc.predict(X_train)
prediction3 = gbc.predict(X_test)
print accuracy_score(prediction3, Y_test)

rfc.fit(X_train, Y_train)
print "RFC"
prediction4_train = rfc.predict(X_train)
prediction4 = rfc.predict(X_test)
print accuracy_score(prediction4, Y_test)

# Combining GBC, RFC, ETC

# Training on combinations
combined_predictions_train = \
    np.array([
        #[prediction1_train[a], prediction2_train[a], prediction3_train[a], prediction4_train[a]] \
        [prediction2_train[a], prediction3_train[a], prediction4_train[a]] \
        for a in range(len(prediction1_train))
    ])

combined_predictions = \
    np.array([
        #[prediction1[a], prediction2[a], prediction3[a], prediction4[a]] \
        [prediction2[a], prediction3[a], prediction4[a]] \
        for a in range(len(prediction1))
    ])

# Voting on predictions
vote_predictions = \
    np.array([
        #[prediction1[a], prediction2[a], prediction3[a], prediction4[a]].count(1) >= 2
        [prediction2[a], prediction3[a], prediction4[a]].count(1) >= 2
        for a in range(len(prediction1))
    ])

#temp = [(combined_predictions_train[a], Y_train[a]) for a in range(len(Y_train))]

print "Combined train"
print " Naive Bayes"
print accuracy_score(BernoulliNB().fit(combined_predictions_train, Y_train).predict(combined_predictions), Y_test)

print "Trees"
print accuracy_score(DTC().fit(combined_predictions_train, Y_train).predict(combined_predictions), Y_test)
print accuracy_score(SVC().fit(combined_predictions_train, Y_train).predict(combined_predictions), Y_test)
print accuracy_score(GBC().fit(combined_predictions_train, Y_train).predict(combined_predictions), Y_test)
print accuracy_score(RFC().fit(combined_predictions_train, Y_train).predict(combined_predictions), Y_test)
print accuracy_score(ETC().fit(combined_predictions_train, Y_train).predict(combined_predictions), Y_test)
print accuracy_score(VTC(
    estimators=[
        ('1', dtc_min_samples_leaf),
        ('2', etc),
        ('3', gbc),
        ('4', rfc)
    ],
    voting='soft'
).fit(combined_predictions_train, Y_train).predict(combined_predictions), Y_test)

print "Combined vote"
print accuracy_score(vote_predictions, Y_test)

## Test submissions
'''
p1_train = GBC().fit(X,Y).predict(X)
p2_train = RFC().fit(X,Y).predict(X)
p3_train = ETC().fit(X,Y).predict(X)

p1 = GBC().fit(X,Y).predict(test_data)
p2 = RFC().fit(X,Y).predict(test_data)
p3 = ETC().fit(X,Y).predict(test_data)

combined_predictions_train = \
    np.array([
        [p1_train[a], p2_train[a], p3_train[a]] \
        for a in range(len(p1_train))
    ])

combined_predictions = \
    np.array([
        [p1[a], p2[a], p3[a]] \
        for a in range(len(p1))
    ])

print combined_predictions
print
print Y
output = BernoulliNB().fit(combined_predictions_train, Y).predict(combined_predictions)

vote_predictions = \
    np.array([
        [p1[a], p2[a], p3[a]].count(1) >= 2
        for a in range(len(p1))
    ])
output = GBC().fit(X, Y).predict(test_data)
'''

# create_submission('results_GBC.txt', output)

inpFile.close()
