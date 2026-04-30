import numpy as np
import os
import pandas as pd
from pandas import DataFrame 
from keras.utils.np_utils import to_categorical
os.environ['KERAS_BACKEND']='tensorflow'
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Flatten, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, Dropout, Activation,GRU
from keras.models import Model, Sequential
import keras.utils
from keras import utils as np_utils
#############################################################################
path = os.getcwd()
files = os.listdir(path)
files_xls = [f for f in files if f[-4:] == 'xlsx']
df = pd.DataFrame()
for f in files_xls:
    data = pd.read_excel(f, 'Sheet1')
    data= data.iloc[:,1:2]# letters data.iloc[:,0:1]    
    data=data.T
    df = df.append(data)
    df=df.replace("'", 0)

    df = df.replace(np.nan, 0)
    
    data=np.array(df)
    ############
    ReadExcel2 = pd.read_excel (r'D:\1PhD Thesis\1st Contribution\1st Contribution\7. All Datasets\LabelsNum_v2.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
df2 = DataFrame(ReadExcel2) #add as many columns as you need
labels=df2['labels'].values
labels=np.array(labels)
labels = to_categorical(np.asarray(labels))
labels=labels[:,1:6]
############################################################################
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)
############################################################################
model = Sequential()
model.add(Embedding(input_dim = 895,output_dim=64))
#model.add(Dense(512,activation='relu', input_dim=895))

#model.add(Dense(512,activation='relu'))
model.add(Conv1D(64, kernel_size = 5, activation = 'relu'))
model.add(MaxPooling1D())
model.add(LSTM(100, activation = 'relu',return_sequences=True))
model.add(LSTM(100, activation = 'relu', return_sequences=True))
model.add(Dense(128,activation='relu'))

model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',
 
              optimizer='adadelta',
 
              metrics=['acc'])


# Train the model, iterating on the data in batches of 32 samples
model.fit(x_train,y_train, epochs=20, batch_size=28)
#X_train, x_test, y_train, y_test  = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %f' % (accuracy*100))