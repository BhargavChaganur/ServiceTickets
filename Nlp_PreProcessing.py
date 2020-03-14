# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:08:05 2019

@author: cbbha
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

df = pd.read_csv('final_Bhargav.csv')


from nltk.stem import PorterStemmer
ps = PorterStemmer()
ps.stem(df['Short Description'])

df['stemming'] = df['Short Description'].apply(ps.stem())

def stem_sentences(sentence):
    tokens = sentence.split()
    print(tokens)
    stemmed_tokens = [ps.stem(token) for token in tokens]
    #return ' '.join(stemmed_tokens)
    print(stemmed_tokens)
    return stemmed_tokens

df['stemm'] = df['Short Description'].apply(stem_sentences)

df['stemm'] = df['tokenized_text'].apply(stem_sentences)


df['tokenized_text'] = df['stemm'].apply(word_tokenize)

from nltk.stem import WordNetLemmatizer 
wlem = WordNetLemmatizer()
wlem.lemmatize(df['stemm'])

def lemm(sentence):
    
    lemmetized_tokens = [wlem.lemmatize(token) for token in sentence]
    return lemmetized_tokens

df['lemm'] = df['tokenized_text'].apply(lemm)

from nltk.corpus import stopwords
stoplist = stopwords.words('english') 

cleanwordlist = [word for word in text.split() if word not in stoplist]

def cleanword(list):
    cleanwords = [word for word in list if word not in stoplist]
    return cleanwords

df['clean'] = df['lemm'].apply(cleanword)


freq_dist = nltk.FreqDist(df_list)
rarewords = freq_dist.keys()[-50:] 

import itertools

x=[itertools.chain(df_values)]

df_list = list(itertools.chain.from_iterable(df_values))






