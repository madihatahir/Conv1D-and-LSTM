from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Conv1D, MaxPooling1D, Concatenate, Dropout,LSTM,GRU,BatchNormalization, ReLU
from keras.layers import Flatten, Input
from keras.layers.embeddings import Embedding
import pandas as pd
from pandas import DataFrame 
import numpy as np
import os
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
#####################################################################################
ReadExcel1= pd.read_excel (r'D:\1PhD Thesis\1st Contribution\1st Contribution\7. All Datasets\TextData_v2.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
df1 = DataFrame(ReadExcel1) #add as many columns as you need
texts=df1['Data'].values

ReadExcel2 = pd.read_excel (r'D:\1PhD Thesis\1st Contribution\1st Contribution\7. All Datasets\LabelsNum_v2.xlsx') #for an earlier version of Excel, you may need to use the file extension of 'xls'
df2 = DataFrame(ReadExcel2) #add as many columns as you need
labels=df2['labels'].values
labels=np.array(labels)
docs=texts
labels = to_categorical(np.asarray(labels))
labels=labels[:,1:6]
######################################################################################

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
texts = []
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
indices = np.arange(data.shape[0]) 
np.random.shuffle(indices) 
data = data[indices] 
labels = labels[indices] 
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0]) 
x_train = data[:-nb_validation_samples] 
y_train = labels[:-nb_validation_samples] 
x_val = data[-nb_validation_samples:] 
y_val = labels[-nb_validation_samples:] 
GLOVE_DIR = "C:\\Users\Madiha\Anaconda3" 
embeddings_index = {} 
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf-8") 
for line in f:   
    values = line.split() 
    word = values[0] 
    coefs = np.asarray(values[1:], dtype='float32') 
    embeddings_index[word] = coefs 
f.close() 
print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index)) 
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM)) 
for word, i in word_index.items(): 
    embedding_vector = embeddings_index.get(word) 
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(len(word_index) + 1, 
                            EMBEDDING_DIM,
                             weights=[embedding_matrix],
                             input_length=MAX_SEQUENCE_LENGTH,
                             trainable=True)
 
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') 
embedded_sequences = embedding_layer(sequence_input)
l_cov1= Conv1D(filters=256, kernel_size=3, activation='relu')(embedded_sequences)
l_drop1=Dropout(0.2)(l_cov1)
l_cov2 = Conv1D(filters=128,kernel_size= 3, activation='relu')(l_drop1)
l_drop2=Dropout(0.2)(l_cov2) 
l_cov3 = Conv1D(filters=64, kernel_size=3, activation='relu')(l_drop2)
l_drop3=Dropout(0.2)(l_cov3)
l_cov4 = Conv1D(filters=32, kernel_size=3, activation='relu')(l_drop3)
l_drop4=Dropout(0.2)(l_cov4) 
l_dense = Dense(256, activation='relu')(l_drop4)
l_flat = Flatten()(l_dense) 
preds = Dense(5, activation='softmax')(l_flat) 
model_1 = Model(sequence_input, preds) 
model_1.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc']) 
print("model fitting - Training and Validation")
model_1.summary() 
print (x_train)
print (y_train) 
model_1.fit(x_train, y_train, validation_data=(x_val, y_val), 
          nb_epoch=2, batch_size=128)   
eeX_train, x1_test, eey_train, y1_test  = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
loss, accuracy1 = model_1.evaluate(x1_test, y1_test)
print('Accuracy on Text: %f' % (accuracy1*100))

######################################################################################
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
labels=df2['Labels'].values
labels=np.array(labels)
labels = to_categorical(np.asarray(labels))
labels=labels[:,1:6]
############################################################################
data = data.reshape((data.shape[0], data.shape[1], 1))
indices = np.arange(data.shape[0]) 
np.random.shuffle(indices) 
data = data[indices] 
labels = labels[indices] 
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0]) 
x2_train = data[:-nb_validation_samples] 
y2_train = labels[:-nb_validation_samples] 
x2_val = data[-nb_validation_samples:] 
y2_val = labels[-nb_validation_samples:] 
############################################################################
# define model
model_2 = Sequential()
model_2.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(895, 1)))
model_2.add(Dropout(0.2))
model_2.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model_2.add(Dropout(0.2))
model_2.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model_2.add(Dropout(0.2))
model_2.add(Dense(256, activation='relu'))
model_2.add(Dropout(0.2))
model_2.add(Flatten())
model_2.add(Dense(5, activation='softmax'))
model_2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['acc'])
model_2.summary()
model_2.fit(x2_train,y2_train, validation_data=(x2_val, y2_val), 
          nb_epoch=2, batch_size=128)
eX_train, x2_test, ey_train, y2_test  = train_test_split(x2_train, y2_train, test_size=0.2, random_state=1)
loss, accuracy2 = model_2.evaluate([x2_test],[y2_test])
print('Accuracy on keysstrokes: %f' % (accuracy2*100))
######################################################################################

from keras.models import Model
from keras.layers import concatenate

merged_layers = concatenate([model_1.output, model_2.output])

L1=(Dense(256, activation='relu'))(merged_layers)
L2=(Dropout(0.2))(L1)
L3=(Dense(128, activation='relu'))(L2)
L4=(Dropout(0.2))(L3)
L5=(Dense(64, activation='relu'))(L4)
L6=(Dropout(0.2))(L5)

out = Dense(5, activation='softmax')(L6)
merged_model = Model([model_1.input, model_2.input], [model_1.output,model_2.output])
merged_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
merged_model.fit([x_train,x2_train],y2_train, validation_data=([x_val,x2_val], y2_val), 
          nb_epoch=2, batch_size=128)
merged_model.summary()
loss, accuracy = merged_model.evaluate([x1_test,x2_test],[y1_test])

print('Accuracy using merged model: %f' % (accuracy*100))
#loss, accuracy = merged_model.evaluate(x2_test, y2_test)
#print('Accuracy on Keystrokes: %f' % (accuracy*100))
"""
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#merged = Model(inputs=[model_1.input, model_2.input],outputs=[[model_1.output, model_2.output]])
#merged.summary()
"""