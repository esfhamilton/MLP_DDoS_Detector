'''
@Title: MLP DDoS Detector
@Created: 06/02/2021
@Last Modified: 01/05/2021
@Author: Ethan Hamilton

MLP based models used to classify DoS/ DDoS attacks from network flows
in the CSE-CIC-IDS-2018 and CIC-DDoS2019 datasets.  
'''

import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.regularizers import l2
from keras.utils import np_utils
from keras import metrics
import matplotlib.pyplot as plt 
import statistics

# Comment out if using CPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

# Used to test 2018 and 2019 datasets with no excluded attacks 
##df = pd.read_csv('Datasets/2018_Preprocessed.csv',sep='\s*,\s*',engine='python')
##df = pd.read_csv('Datasets/2019_Preprocessed.csv',sep='\s*,\s*',engine='python')
##y = np.asarray(df['Label'])
##X = np.asarray(df.drop(columns=['Label']))
##X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15)

# Used to test excluded attacks
dfTrain = pd.read_csv('Datasets/train.csv',sep='\s*,\s*',engine='python')
dfTest = pd.read_csv('Datasets/test.csv',sep='\s*,\s*',engine='python')
dfTrain = dfTrain.sample(frac=1)
dfTest = dfTest.sample(frac=1)
y_train = np.asarray(dfTrain['Label']) 
X_train = np.asarray(dfTrain.drop(columns=['Label']))
y_test = np.asarray(dfTest['Label'])
X_test = np.asarray(dfTest.drop(columns=['Label']))

# Ranges used for hyperparameter tuning
hiddenRange = [8,16,24,32,40,48,56,64] # Default = 24
optimizers = [Adam, SGD, RMSprop, Adamax, Nadam, Ftrl]
etaRange = [1,0.1,0.01,0.001,0.0001,0.00001] # Default 0.001
batchRange = [4,8,16,32,64] # Default 32
epochRange = [1,3,5,10,20,30,40,50] # Default 1
f1Scores = [[0.0]*len(hiddenRange) for _ in range(len(hiddenRange))] 

acc = []
pre = []
rec = []
f1S = []

repeats = 10
# THIS MODEL HAS BEEN TUNED ON THE 2018 DATASET
for k in range(repeats):
        inp = Input(shape=(58,))
        hidden_1 = Dense(56, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inp)
        hidden_2 = Dense(8, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(hidden_1)
        out = Dense(1, activation='sigmoid')(hidden_2)
        model = Model(inputs=inp, outputs=out)
        opt = keras.optimizers.Nadam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy','TruePositives','TrueNegatives','FalsePositives','FalseNegatives'])
        batch_size = 32 
        num_epochs = 10 
	
        model.fit(X_train, y_train, 
                          batch_size=batch_size, epochs=num_epochs,
                          verbose=1, validation_split=0.1) 


        y_pred = model.predict(X_test)
        print("PREDICTIONS:")
        print(y_pred)
        acc.append(accuracy_score(y_test,tf.greater(y_pred,.5)))
        pre.append(precision_score(y_test,tf.greater(y_pred,.5)))
        rec.append(recall_score(y_test,tf.greater(y_pred,.5)))
        f1S.append(f1_score(y_test,tf.greater(y_pred,.5)))

print("Accuracies: {}".format(acc))
print("Precisions: {}".format(pre))
print("Recalls: {}".format(rec))
print("F1 Scores: {}".format(f1S))

print("Averages:")
print("Acc: {}".format(statistics.mean(acc)))
print("Pre: {}".format(statistics.mean(pre)))
print("Rec: {}".format(statistics.mean(rec)))
print("F1: {}".format(statistics.mean(f1S)))
print("Maximums:")
print("Acc: {}".format(max(acc)))
print("Pre: {}".format(max(pre)))
print("Rec: {}".format(max(rec)))
print("F1: {}".format(max(f1S)))


# THIS MODEL HAS BEEN TUNED ON THE 2019 DATASET
for k in range(repeats): 
	inp = Input(shape=(58,)) 
	hidden_1 = Dense(56, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inp) 
	hidden_2 = Dense(40, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(hidden_1) 
	out = Dense(1, activation='sigmoid')(hidden_2) 
	model = Model(inputs=inp, outputs=out)
	opt = keras.optimizers.Nadam(learning_rate=0.001) 

	model.compile(loss='binary_crossentropy', 	

				  optimizer=opt, 			              
				  metrics=['accuracy','TruePositives',
				  'TrueNegatives','FalsePositives',
				  'FalseNegatives']) 		

	batch_size = 4
	num_epochs = 40

	model.fit(X_train, y_train, 
			  batch_size=batch_size, epochs=num_epochs,
			  verbose=1, validation_split=0.1) 

	y_pred = model.predict(X_test)
	acc.append(accuracy_score(y_test,tf.greater(y_pred,.5)))
	pre.append(precision_score(y_test,tf.greater(y_pred,.5)))
	rec.append(recall_score(y_test,tf.greater(y_pred,.5)))
	f1S.append(f1_score(y_test,tf.greater(y_pred,.5)))

print("Accuracies: {}".format(acc))
print("Precisions: {}".format(pre))
print("Recalls: {}".format(rec))
print("F1 Scores: {}".format(f1S))


print("Averages:")
print("Acc: {}".format(statistics.mean(acc)))
print("Pre: {}".format(statistics.mean(pre)))
print("Rec: {}".format(statistics.mean(rec)))
print("F1: {}".format(statistics.mean(f1S)))
print("Maximums:")
print("Acc: {}".format(max(acc)))
print("Pre: {}".format(max(pre)))
print("Rec: {}".format(max(rec)))
print("F1: {}".format(max(f1S)))


# Hyperparameter Plots
plt.plot(epochRange, f1Scores, 'r', label="2018 Dataset")
plt.plot(epochRange, f1Scores2, 'b', label="2019 Dataset")
plt.xticks(epochRange, epochRange)
plt.ylabel('F1-Score')
plt.xlabel('Epochs')
plt.legend()
plt.show()  

#model.summary() 
