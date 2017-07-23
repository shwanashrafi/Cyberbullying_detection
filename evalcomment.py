# -*- coding: utf-8 -*-
#import matplotlib.pyplot as plt
import sys, re, string, os
import pandas as pd
import numpy as np
import pickle
import operator
from collections import Counter
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import tensorflow as tf

with open("mytokenizer", "rb") as f:
   tokenizer = pickle.load(f)

# load saved model
json_file = open('commentcleanermodel2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("commentcleanermodel2.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
loaded_model._make_predict_function()
graph = tf.get_default_graph()


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



def classifier(cmnt, tk, mdl):

   cmnt = cmnt.encode('utf-8')
   comment = [cleanComment(cmnt)]
   tk.fit_on_texts(comment)
   sequences = tk.texts_to_sequences(comment)
   testData = pad_sequences(sequences, maxlen=100)
   with graph.as_default():
     labeltarget = mdl.predict(testData)
      
   return labeltarget > .5



