# Multiple Inputs
#from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
# first input model
model = Sequential()
model.add(Embedding(2000, 300, input_length=100))
model.add(Conv1D(32, kernel_size=4, activation='relu'))
model.add( Dropout(0.2))
model.add( Conv1D(16, kernel_size=4, activation='relu'))
model.add( Dropout(0.2))
model.add( Flatten())
# second input model
visible2 = Input(shape=(895,1))
conv21 = Conv1D(32, kernel_size=4, activation='relu')(visible2)
pool21 = Dropout(0.2)(conv21)
conv22 = Conv1D(16, kernel_size=4, activation='relu')(pool21)
pool22 = Dropout(0.2)(conv22)
flat2 = Flatten()(pool22)
# merge input models
merge = concatenate([model.output, flat2])
# interpretation model
hidden1 = Dense(10, activation='relu')(merge)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(5, activation='softmax')(hidden2)
model = Model(inputs=[model.input, visible2], outputs=output)
# summarize layers
print(model.summary())
# plot graph
#plot_model(model, to_file='multiple_inputs.png')
import pandas as pd
from pandas import DataFrame 
import numpy as np
import os
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
ReadExcel1= pd.read_excel (r'D:\PhD Thesis\TextData.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
df1 = DataFrame(ReadExcel1) #add as many columns as you need
texts=df1['Data'].values
ReadExcel2 = pd.read_excel (r'D:\PhD Thesis\KS_Labels.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
df2 = DataFrame(ReadExcel2) #add as many columns as you need
labels=df2['Labels'].values
labels=np.array(labels)
docs=texts
labels = to_categorical(np.asarray(labels))
labels=labels[:,1:6]
tokenizer = Tokenizer(nb_words=2000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
texts = []
data1 = pad_sequences(sequences, maxlen=100)
############################################################################################################
############################################################################################################
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
    data2=np.array(df)
data2 = data2.reshape((data2.shape[0], data2.shape[1], 1));
############################################
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
x1_train, x1_test, y1_train, y1_test  = train_test_split(data1, labels, test_size=0.3, random_state=1)
x2_train, x2_test, y2_train, y2_test  = train_test_split(data2, labels, test_size=0.3, random_state=1)
model.fit([x1_train,x2_train],[y1_train], nb_epoch=200, batch_size=128)
loss, accuracy = model.evaluate([x1_test,x2_test],[y1_test])