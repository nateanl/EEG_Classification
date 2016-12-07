import csv
import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
features= []
label = []
feature = []
with open('./EEG_data.csv','rb') as f:
    reader = csv.reader(f)
    i=0
    for row in reader:
        if(i==0):
            i+=1
            print row
            continue
        for i in range(0,len(row)):
            row[i] = float(row[i])
        features.append(row[0:14])
        label.append(row[14])

X = np.asarray(features)
Y = np.asarray(label)
features = {}
output = {}
for i in range(X.shape[0]):
    tu = int(X[i][0]*10 + X[i][1])
    if tu not in features.keys():
        features[tu] = X[i][2:14]
    elif features[tu].shape[0]<1344:
        features[tu] = np.concatenate((features[tu],X[i][2:14]),axis =0)
    output[tu]= Y[i]
input = np.zeros((100,1344),dtype = float)
labels = np.zeros((100,1),dtype = int)
for i in features.keys():
    input[i,:] = features[i]
    labels[i] = output[i]

print "Begin SVM model"
svc = svm.SVC(C=1, kernel='sigmoid')
print np.mean(cross_val_score(svc, input,labels[:,0], cv=5, n_jobs=-1))


#accuracy result
#rbf SVM: 0.512606348903
#linear SVM: 0.671704260652
#simoid SVM: 0.510025062657