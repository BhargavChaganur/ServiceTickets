# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:33:39 2019

@author: cbbha
"""

import logging
import pandas as pd
import numpy as np
from numpy import random
#import gensim
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

print(df['Description'].apply(lambda x: x.split(' ')))

def print_plot(index):
    example = df[df.index == index][['Description', 'Type']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Type:', example[1])
        

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


df['Description_2'] = df['Deescription_1'].apply(clean_text)

df["Deescription_1"]= df["Description"].astype(str)

df['Description_2'].apply(lambda x: len(x.split(' '))).sum()

X = df.Description_2
y = df.Type
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)



from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

%%time
from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

x = 'Need access for SAP N93'


y_pred_test = nb.predict()

df_test = pd.read_excel('Test_1.xlsx',header = None)

from sklearn.metrics import precision_recall_fscore_support


from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)

%%time

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))


