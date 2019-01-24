import tensorflow as tf
from keras.preprocessing import sequence
import time
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.metrics import confusion_matrix, precision_score, recall_score, cohen_kappa_score
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
from sklearn import preprocessing
import pandas as pd
from keras.optimizers import SGD, Adam, Nadam, Adagrad
from sklearn.model_selection import StratifiedKFold
from keras.layers import BatchNormalization
import pandas as pd
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam, Nadam
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from keras.utils.vis_utils import plot_model
from keras.layers.normalization import BatchNormalization
import keras
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import keras_metrics
from sklearn.model_selection import StratifiedKFold
import time


#def load_data(input_file):
def train_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train,batch_size=32, nb_epoch=1000)
    return model.evaluate(x_test, y_test)


df = pd.read_csv("clean_data.csv")
X = np.array(df.drop(['health'], 1)) 
df.dropna(inplace=True)
y = np.array(df['health'])


Y = to_categorical(y)
print(Y)
X_train, X_test, y_train, y_test =train_test_split (X, Y, test_size=0.2)


print("categorized target = ",Y)

print('y_test = ',y_test)

max_features = 1000000
maxlen = 202
embedding_size = 320






batch_size = 10


epochs = 20


def create_model():
  # def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(512, input_dim=34, init='uniform'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, init='uniform'))
    model.add(Activation('softmax'))

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = Adam(lr=0.05)


    print ('Compiling...')
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  #metrics=['accuracy'])
                  metrics=['accuracy'])



  
    return model





print(X_train.shape)

model = create_model()

print ('Fitting model...')
history = model.fit(X_train, y_train, batch_size=100, nb_epoch=1000, validation_split = 0.1, verbose = 1)


# plt.show()
score, acc = model.evaluate(X_test, y_test, batch_size=100)
y_pred = model.predict(X_test)
print('acc = ', acc)
print(y_pred)
y_pred = (y_pred > 0.5) 


print(classification_report(y_test, y_pred))

print(y_pred)
model.summary()

