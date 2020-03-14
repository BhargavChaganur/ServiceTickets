# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Importing Libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import itertools
#Reading data
df = pd.read_excel('DUMP.xlsx',sheet_name = 'Req')
#Converting words to lower case
df = df.apply(lambda x: x.astype(str).str.lower())
#Tokenizing data
df['tokenized_data'] = df['Description'].apply(word_tokenize)

#Converting dataframe to list
list_1 = df['tokenized_data'].values

list2 = list(itertools.chain.from_iterable(list_1))


#Stop Words
stoplist = stopwords.words('english')

#Customizing stopwords
stopword_custom = ['dear',':','.',')','>','?','<','/','&','#','--','@','(','thanks','regards','thank','you','thankyou','please','kindly','-','sent','subject',',','team']
stopword_custom_1 = ['henkel.com',';','2019','cc','[',']','!','1','~henkelsd_english']
stopword_custom_2 = ['49',	'hi',	'pm',	'may',	'recipient',	'unisys.com',	'phone',	'ext',	'accenture.com',	'phone',	'help',	'-',	'attachments',	'henkelsdenglish',	'use',	'2',	'immediately',	'www.henkel.com',	"'",	'hello',	'|',	'düsseldorf',	'de',	'//www.henkel.com',	'freundlichen',	'loctite',	'co.',	's',	'grüßen',	'tuesday',	'wednesday',	'monday',	'',	'2018',	'also',	'49',	'3',	'friday',	'4',	'...',	'pls',	'na',	'8',	'91',	'6',	'________________________________________',	'86',	'20',	'july',	'1',	'skype',	'requester',	'7',	'troubleshooting',	'b',	'11',	'30',	'global.accenture.appl.authorizations.m',	'15',	'100',	'22',	'_____________________________',	'0',	'f',	'19','421',	'______________________________',	'18',	'8230',	'9',	'1',	'25',	'14',	'____________________________________',	'k',	'1',	'n',	'pir',	'34',	'_',	'_________________________________',	'*',	'_______________________________',	'adresátom',	'adresátovi',	'akejko¾vek',	'chránené',	'deleted',	'dôverné',	'elektronická',	'eub1',	'inej',	'informujete',	'informácie',	'kopírovanie',	'ktoré',	'môže',	'môžu',	'neoprávnené',	'neoprávnený',	'následne',	'obsahom',	'odosielate¾a',	'odpoveïou',	'okamžite',	'osoby',	'pod¾a',	'považované',	'predpisov',	'protiprávne',	'právnych',	'prípade',	'prístup',	'rozhodnutie',	'rozširovanie',	'správa',	'správu',	'táto',	'túto',	'urèená',	'vymažte',	'výhradne',	'zamýš¾aným',	'44',	'___________________________________',	'65',	'n/a',	'n.',	'r.o',	'sjhf',	'1st',	'33']
stoplist.extend(stopword_custom_2)

#Cleanword List
cleanwordlist = [word for word in list2 if word not in stoplist]

#Frequency Distribution
freq_dist = nltk.FreqDist(cleanwordlist)

freq_dist.most_common(50)

















## Testing
print(freq_dist.












