# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import sys, re, string
import numpy as np
import nltk
import operator
import pandas as pd


def removePunc( post ):

    post = re.sub('newline_token', ' ', post)
    punctMarks = r"""!@#$%^&*()-_:;"<>,.//?|\{}[]+=`~£"""
    postRemovedPunc = string.maketrans( punctMarks, ' '*len(punctMarks) )
    postRemovedPunc = post.translate( postRemovedPunc )
    postRemovedPunc = re.sub( r"\s'|'\s|^'|'$", " ", postRemovedPunc )
    postRemovedPunc = re.sub( "'", "_", postRemovedPunc )
    
    return postRemovedPunc

def cleanPost( post ):
    
    words = removePunc( post.lower() ).split()
    if post.isspace() or len(words) == 0:
	    return False, None
    return True, words

class Ngram:

    def __init__( self, datalist, labellist, nngrams, bullyWordProbThrsh ):
        self.nngrams = nngrams
        self.bullyWordProbThrsh   = bullyWordProbThrsh;
        self.bullyWordFreq, self.normalWordFreq   = self.getWordFreq( datalist, labellist ) 

    def getWordFreq( self, datalist, labellist ):
       freqBully =  nltk.FreqDist()
       freqNormal = nltk.FreqDist()
 
       for i  in range(len(datalist)):
         validPost = datalist[i][0]
         cleanedPost = datalist[i][1]
       
         if not validPost:
           continue
	   
         ngrams = nltk.ngrams(cleanedPost, self.nngrams)
         for ngram in ngrams:
           ngramtmp = " ".join( ngram )
           if int(labellist[i]) == 1:
             freqBully[ngramtmp] += 1 
           else:
             freqNormal[ngramtmp] += 1 
       return freqBully, freqNormal


    def getMessageRank( self, words ):
        ngrams = nltk.ngrams( words, self.nngrams)
	isBullyPost = False
	maxMsgRank = 0.0
        for ngram in ngrams:
	    ngramtmp = " ".join( ngram )
	    bullyWordProb   = 1.0 * self.bullyWordFreq[ngramtmp] / self.bullyWordFreq.N()     
            normalWordProb  = 1.0 * self.normalWordFreq[ngramtmp] / self.normalWordFreq.N()  
       	    bullyWordRank   = bullyWordProb * self.bullyWordProbThrsh
            normalWordRank  = normalWordProb * (1 - self.bullyWordProbThrsh)
            if  bullyWordRank + normalWordRank == 0:
	        continue
            msgRank = bullyWordRank / ( bullyWordRank + normalWordRank )
            if ( msgRank > maxMsgRank ):
                maxMsgRank = msgRank
        
        if ( maxMsgRank > 0.8 ):
            isBullyPost = True
 
        return isBullyPost, maxMsgRank

commentfile = '../data/attack_annotated_comments.tsv'
labelfile = '../data/attack_annotations.tsv'
badwWordProbThrsh = .5
df = pd.read_csv(commentfile, sep='\t', usecols=[0,1])
# clean all the comments
df.loc[:,'comment'] = df.loc[:,'comment'].apply(cleanPost)
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
datalist = list(dfAll['comment'])
labellist = list(dfAll['attack'])

ngram = []
for n in range(2):

	ngram.append( Ngram( datalist, labellist, n + 1, badwWordProbThrsh ) )


	num_word = 50
	mostcommonwords = dict(ngram[n].bullyWordFreq.most_common(num_word))
	mostcommonwords_sorted = sorted(mostcommonwords.items(), key=operator.itemgetter(1), reverse=True)
	plt.bar(range(len(mostcommonwords)), [w[1] for w in mostcommonwords_sorted])
	plt.xticks(range(len(mostcommonwords)), [w[0] for w in mostcommonwords_sorted], rotation=90)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
	plt.ylabel('Count', fontsize=14)
	plt.title('Frequency of the ' + str(num_word) + ' most common ' + str(n+1) + '-grams', fontsize=14)
	plt.show()

for c in range(2):
	for n in range(2): 
	    isBullyAttack = []
	    PostRank = []
	    classificationErr = []
	    count=0
	    for i in range(len(datalist)):
		if int(labellist[i]) == c:
		    words = datalist[i][1]
		    b, r = ngram[n].getMessageRank( words )
		    isBullyAttack.append(b)
		    PostRank.append(r)
		    classificationErr.append(1 - int(int(b) == int(labellist[i])))
		    count += 1
		    if count == 50:
		       break
	    plt.subplot(2,1,c+1)
            if n == 0:
	        plt.plot(range(len(PostRank)), PostRank, linestyle='None', marker='o', label=str(n+1)+'-grams')
            else:
	        plt.plot(range(len(PostRank)), PostRank, linestyle='None', marker='s', label=str(n+1)+'-grams')
 
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)


        plt.legend()
        if c == 1: plt.xlabel('Comment', fontsize=14);
        plt.ylabel('Rank of a Comemnt', fontsize=14)
        if c == 0:
            plt.title('Good Comments', fontsize=14);
        else:
            plt.title('Bad Comments', fontsize=14);

plt.show()

