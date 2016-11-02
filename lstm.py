
# coding: utf-8

# In[1]:

import csv
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier


# In[2]:

features = []
label = []
feature = []


# In[3]:

with open('/Users/Near/Documents/EEG_Classification/EEG_data.csv','rb') as f:
    reader = csv.reader(f)
    i=0
    for row in reader:
        if(i==0):
            i+=1
            print row
            continue
        for i in range(0,len(row)):
            row[i] = float(row[i])
        features.append(row[2:14])
        label.append(row[14])


# In[4]:

X = np.asarray(features)
Y = np.asarray(label)


# In[5]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, label, test_size=0.5, random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


# In[6]:

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


# In[7]:

X_train.shape


# In[8]:

neigh = KNeighborsClassifier(n_neighbors=100)
neigh.fit(X_train, y_train)
print neigh.score(X_test, y_test)


# In[9]:

from sklearn.linear_model import SGDClassifier


# In[10]:

clf = SGDClassifier(loss="hinge", penalty="l2")


# In[11]:

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print clf.score(X_test, y_test)


# In[12]:

import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file


# In[13]:

data_dim = 12
timesteps = 1
batch_size=1
model = Sequential()
model.add(LSTM(24, batch_input_shape= (1,1,12)))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='binary_crossentropy')


# In[14]:

new_input = np.zeros((12811,1,12))
for i in range(0,12811):
    for j in range(0,12):
        new_input[i,:,j]= X[i,j]


# In[15]:

new_output = np.zeros((12811,1))
for i in range(0,12811):
    new_output[i,0]= Y[i,]


# In[ ]:

X.shape


# In[ ]:

model.fit(new_input, new_output,
          batch_size=batch_size, nb_epoch=5
          )


# In[ ]:

y_test.shape


# In[ ]:

new_test = np.zeros((6406,1,12))
for i in range(0,6406):
    for j in range(0,12):
        new_test[i,:,j]= X_test[i,j]


# In[ ]:

prediction = model.predict_classes(new_input, batch_size=1)


# In[ ]:

prediction


# In[ ]:

count =0;
for i in range(12811):
    if prediction[i]==new_output[i]:
        count+=1


# In[ ]:

6748/12811.0


# In[ ]:

new_output=new_output.reshape(1281,5)
new_input=new_input.reshape(1281,5,12)


# In[ ]:

data_dim = 12
timesteps = 5
batch_size=1
model2 = Sequential()
model2.add(LSTM(12, return_sequences=False,
batch_input_shape=(batch_size, timesteps, data_dim)))
#model2.add(Activation('softmax'))
model2.compile(optimizer='rmsprop', loss='mse')


# In[ ]:

model2.fit(new_input, new_output,
          batch_size=batch_size, nb_epoch=5
          )


# In[ ]:

prediction = model2.predict(new_input, batch_size = 1)


# In[ ]:

for i in range(1281):
    for j in range(5):
        if prediction[i,j]>0.5:
            prediction[i,j]=1
        else:
            prediction[i,j]=0


# In[ ]:

count=0
for i in range(1281):
    for j in range(5):
        if prediction[i,j]== new_output[i,j]:
            count+=1
count


# In[ ]:

6405-3167


# In[ ]:

3238/6405.0


# In[ ]:



