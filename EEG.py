import csv
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
features = []
label = []
with open('EEG_data.csv','rb') as f:
    reader = csv.reader(f)
    i=0
    for row in reader:
    	if(i==0):
    		i+=1
    		continue
        features.append(row[2:14])
        label.append(row[14])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, label, test_size=0.5, random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
# clf = svm.SVC()
# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print clf.score(X_test, y_test)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print neigh.score(X_test, y_test)