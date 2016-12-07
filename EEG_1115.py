
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
from keras.layers import Dense, Activation, Dropout
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


# In[ ]:

print len(features)
print X.shape

# In[3]:

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

# In[4]:
input = np.zeros((100,1344),dtype = float)
labels = np.zeros((100,1),dtype = int)
for i in features.keys():
    input[i,:] = features[i]
    labels[i] = output[i]

# new_input = input.reshape(100,112,12)
# new_input = new_input.swapaxes(1,2)
# input = new_input.reshape(100,1,1344)




# print "Begin Convolutional Neural Networks model"
# X_train, X_test, Y_train, Y_test = train_test_split(input, labels, test_size=0.1, random_state=i)
# model = Sequential()
# # first set of CONV => RELU => POOL
# # number of convolutional filters
# n_filters = 2
#
# # convolution filter size
# # i.e. we will use a n_conv x n_conv filter
# n_conv = 2
# # pooling window size
# # i.e. we will use a n_pool x n_pool pooling window
# n_pool = 2
# model.add(Convolution2D(
#         n_filters, n_conv, n_conv,
#
#         # apply the filter to only full parts of the image
#         # (i.e. do not "spill over" the border)
#         # this is called a narrow convolution
#         border_mode='valid',
#
#         # we have a 28x28 single channel (grayscale) image
#         # so the input shape should be (1, 28, 28)
#         input_shape=(1, 112, 12)
# ))
# model.add(Activation('relu'))
#
# model.add(Convolution2D(n_filters, n_conv, n_conv))
# model.add(Activation('relu'))
#
# # then we apply pooling to summarize the features
# # extracted thus far
# model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
# model.add(Dropout(0.25))
#
# # flatten the data for the 1D layers
# model.add(Flatten())
#
# # Dense(n_outputs)
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
#
# # the softmax output layer gives us a probablity for each class
# model.add(Dense(1))
# model.add(Activation('softmax'))
# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
# X_train = X_train.reshape(90,1,112,12)
# X_test = X_test.reshape(10,1,112,12)
# # how many examples to look at during each training iteration
# batch_size = 128
# # how many times to run through the full set of examples
# n_epochs = 100
# # the training may be slow depending on your computer
# model.fit(X_train,
#           Y_train[:,0],
#           batch_size=batch_size,
#           nb_epoch=n_epochs)
# loss, accuracy = model.evaluate(X_test, Y_test)
# print('loss:', loss)
# print('accuracy:', accuracy)
#


print "Begin LSTM model"
accuracy = 0.0
for i in range(0,5):
    # X_train = np.concatenate((input[0:20*(i-1),:,:],input[20*i:100,:,:]),axis=0)
    # Y_train = np.concatenate((labels[0:20*(i-1)],labels[20*i:100]),axis=0)
    # X_test = input[20*(i-1):20*i,:,:]
    # Y_test = labels[20*(i-1):20*i]
    # X_train = X_train.reshape(80,1,1344)
    # X_test = X_test.reshape(20,1,1344)
    X_train, X_test, Y_train, Y_test = train_test_split(input, labels, test_size=0.2, random_state=i)
    X_train = X_train.reshape(80,1,1344)
    X_test = X_test.reshape(20,1,1344)
    np.random.seed(7)
    # create the model
    model = Sequential()
    batch_size = 5
    #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length, dropout=0.2))
    #model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False, input_shape=(1,1344)))
    #model.add(Dropout(0.2))
    #model.add(LSTM(200, return_sequences = False, input_length=1024))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train,nb_epoch=150)
    # Final evaluation of the model
    scores = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    log.write(str(scores[1])+'\n')
    accuracy += scores[1]
print accuracy/5
log.write(str(accuracy/5)+'\n')
log.close() # you can omit in most cases as the destructor will call it
