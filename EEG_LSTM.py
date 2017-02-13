import csv
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf
np.random.seed(1337)  # for reproducibility
from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers import BatchNormalization
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from nolearn.dbn import DBN
from sklearn.metrics import classification_report, accuracy_score
from dbn.tensorflow import SupervisedDBNClassification
from keras.layers.wrappers import Bidirectional
log= open('5_fold_log_1125.txt','w')
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

print "Begin LSTM model"
accuracy = 0.0
for i in range(0,5):
    X_train, X_test, Y_train, Y_test = train_test_split(input, labels, test_size=0.2, random_state=i*0)
    X_train = X_train.reshape(80,112,12)
    X_test = X_test.reshape(20,112,12)
    y_train = np.zeros((80,112),dtype='int')
    y_test = np.zeros((20,112),dtype='int')
    y_train = np.repeat(Y_train,112, axis=1)
    y_test = np.repeat(Y_test,112, axis=1)
    np.random.seed(1)
    # create the model
    model = Sequential()
    batch_size = 20
    #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length, dropout=0.2))
    #model.add(Dropout(0.2))
    model.add(BatchNormalization(input_shape=(112,12),mode =0,axis=2))
    model.add(Bidirectional(LSTM(50, return_sequences=False, input_shape=(112,12)),merge_mode = 'ave'))
    #model.add(Dropout(0.2))
    #model.add(LSTM(200, return_sequences = False, input_length=1024))
    model.add(Dense(112, activation='hard_sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['binary_accuracy'])
    #print(model.summary())
    model.fit(X_train, y_train,nb_epoch=25)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, batch_size = batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write(str(scores[1])+'\n')
    accuracy += scores[1]
print accuracy/5
log.write(str(accuracy/5)+'\n')
log.close() # you can omit in most cases as the destructor will call it

#average accuracy: 0.690000013262