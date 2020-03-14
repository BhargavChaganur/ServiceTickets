# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 23:15:27 2019

@author: cbbha
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 22:00:53 2019

@author: cbbha
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import SpatialDropout1D
from keras.layers import LSTM
from keras.callbacks.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
import logging
import pandas as pd
import numpy as np
from numpy import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
%matplotlib inline

df = pd.read_excel('DUMP.xlsx')
df = df[pd.notnull(df['Type'])]

#print(df['Description'].apply(lambda x: x.split(' ')))
#
#def print_plot(index):
#    example = df[df.index == index][['Description', 'Type']].values[0]
#    if len(example) > 0:
#        print(example[0])
#        print('Type:', example[1])
        

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

df["Deescription_1"]= df["Description"].astype(str)

df['Description_2'] = df['Deescription_1'].apply(clean_text)



#LSTM Model Building
MAX_WORDS = 100000
MAX_LENGTH = 250
EMBEDDING_DIMENSION = 100

#Tokenization 
tokenizer = Tokenizer(num_words=MAX_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Description_2'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

Z = tokenizer.texts_to_sequences(df['Description_2'].values)
Z1 = pad_sequences(Z, maxlen=MAX_LENGTH)
print('Shape of data tensor:', Z1.shape)

#Converting categorocal to numbers
Y = pd.get_dummies(df['Type']).values
print('Shape of label tensor:', Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(Z1,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_WORDS, EMBEDDING_DIMENSION, input_length=Z1.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 4
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

new_ticket = ['Unable to open application']
seq = tokenizer.texts_to_sequences(new_ticket)
padded = pad_sequences(seq, maxlen=MAX_LENGTH)
pred = model.predict(padded)
labels = ['Incident','Request']
print(pred, labels[np.argmax(pred)])














