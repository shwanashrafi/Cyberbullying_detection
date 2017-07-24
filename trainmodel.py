# -*- coding: utf-8 -*-
#import matplotlib.pyplot as plt
import sys, re, string, os
import pandas as pd
import numpy as np
import pickle
#import nltk
import operator
from collections import Counter
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json

def cleanComment( cmnt ):
    comment = re.sub('NEWLINE_TOKEN', ' ', cmnt)
    punctMarks = r"""!@#$%^&*()-_:;"<>,.//?|\{}[]+=`~£"""
    commentRemovedPunc = string.maketrans( punctMarks, ' '*len(punctMarks) )
    commentRemovedPunc = comment.translate( commentRemovedPunc )
    commentRemovedPunc = re.sub( r"\s'|'\s|^'|'$", " ", commentRemovedPunc )
    commentRemovedPunc = re.sub( "'", "_", commentRemovedPunc )
    commentRemovedPunc = commentRemovedPunc.lower()
    if comment.isspace() or len(comment) ==0:
      return "ZZZ"
    return commentRemovedPunc


readlen = 10000000

filename = '../data/attack_annotated_comments.tsv'
labelfile = '../data/attack_annotations.tsv'
df = pd.read_csv(filename, sep='\t', usecols=[0,1])
# clean all the comments
df.loc[:,'comment'] = df.loc[:,'comment'].apply(cleanComment)

lab = pd.read_csv(labelfile, sep='\t', usecols=[0,6])
lab = lab.groupby(['rev_id']).agg({'attack':lambda x: sum(x)>x.count()/2})

lab_pos = lab[lab['attack']>0]
L = int(lab_pos.count())
lab_neg = lab[lab['attack']==0][0:L]
lab = lab_pos.append(lab_neg, ignore_index=False).sample(frac=1)

df = df.set_index('rev_id')
df.index = df.index.astype(str)
lab.index = lab.index.astype(str)
dfAll = lab.join(df)

trainData = list(dfAll['comment'])
trainLabel =  list(dfAll['attack'])

tokenizer = Tokenizer(num_words=15000)
tokenizer.fit_on_texts(trainData)
sequences = tokenizer.texts_to_sequences(trainData)
data = pad_sequences(sequences, maxlen=100)

# create a network
model = Sequential()
model.add(Embedding(15000, 128, input_length=100))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print "Training...\n"
model.fit(data, np.array(trainLabel), validation_split=0.5, epochs=2)



scores = model.evaluate(data, np.array(trainLabel), verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save the tokenizer and model
with open("mytokenizer", "wb") as f:
  pickle.dump(tokenizer, f)


model_json = model.to_json()
with open("commentcleanermodel2.json", "w") as json_file:
    json_file.write(model_json)
#save model in HDF5
model.save_weights("commentcleanermodel2.h5")
