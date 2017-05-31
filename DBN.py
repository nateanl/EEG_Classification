import csv
import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.io as sio
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from nolearn.dbn import DBN
from sklearn.metrics import classification_report, accuracy_score
from dbn.tensorflow import SupervisedDBNClassification
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
print X.shape[1]
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


print "Begin DBN model"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=i)
dbn_model = DBN([X_train.shape[1], 300, 2],
                learn_rates = 0.9,
                learn_rate_decays = 0.1,
                epochs = 100,
                verbose = 1)
dbn_model.fit(X_train, Y_train)
y_true, y_pred = Y_test, dbn_model.predict(X_test) # Get our predictions
print(classification_report(y_true, y_pred)) # Classification on each digit
print 'The accuracy is:', accuracy_score(y_true, y_pred)

print "Begin DBN V2 model"
classifier = SupervisedDBNClassification(hidden_layers_structure=[200,200],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)
# Test
Y_pred = classifier.predict(X_test)
print 'Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred)

