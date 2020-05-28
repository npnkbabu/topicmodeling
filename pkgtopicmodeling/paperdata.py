#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from gensim import corpora, models
from gensim.models import TfidfModel
import gensim
nltk.download('wordnet')
nltk.download('punkt')
#nltk.download()
import sys
nltk.download('stopwords')


# In[4]:


class paperdata:
  __papersRawData = pd.DataFrame()
  __processed_tokens = []
  __stop_words = set(stopwords.words('english'))
  __wordnetLemmatizer = WordNetLemmatizer()
  __stemmer = PorterStemmer()
  gensim_tfidf = []
  id2word = gensim.corpora.Dictionary()
  corpus_tfidf = []
  processeddata = []

  def __init__(self):
    print('starting paperdata')
  
  def loadAndPreprocess(self):
    try:
      self.__papersRawData = pd.read_csv('abcnews-date-text.csv')
      print('Raw data info :')
      self.__papersRawData['index'] = self.__papersRawData.index
      self.__papersRawData = self.__papersRawData[:5000]
      self.__printinfo_raw()
      self.__papersRawData['headline_text_processed'] = self.__papersRawData['headline_text'].map(self.preprocess)
      print(self.__papersRawData['headline_text_processed'][:10])
      self.__processed_tokens = list(self.__papersRawData['headline_text_processed'].values)
      print('Conducting EDA showing wordcloud and most common 10 words in histogram')
      self.__EDA()
      print('preparing the data is completed. Now we need to provide this to model')
    except:
      print('error in __loadAndPreprocess ***************************')
      print(sys.exc_info()[0])


  
  def preprocess(self,line):
    result=[]
    tokens=[]
    try:
      #print('map calling')
      #remove punctuations 
      line = re.sub('[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]','',line)
      line = line.strip()
      #tokenize
      tokens = word_tokenize(line)
      #convert to lower
      tokens = list(map(lambda x : x.lower(), tokens))
      #remove stop words
      tokens = [x for x in tokens if x not in self.__stop_words]
      #stem each word
      tokens = list(map(lambda x : self.__stemmer.stem(x),tokens))
      
      #get POS tagging for each word
      #lemmatize each word with POS tagging
      pos_tags = nltk.pos_tag(tokens)
      for token,postag in pos_tags:    
        pos = self.__getPosTagDef(postag)
        if pos != '':
          result.append(self.__wordnetLemmatizer.lemmatize(token,pos))
        else:
          result.append(self.__wordnetLemmatizer.lemmatize(token))
      #print(result)
    
    except:
      print('error*******************')
      print(tokens)
      print(sys.exc_info())
    return result
  
  def __getPosTagDef(self,postag):
    try:
      if postag.startswith('J'):
          return wordnet.ADJ
      if postag.startswith('V'):
          return wordnet.VERB
      if postag.startswith('N'):
          return wordnet.NOUN
      if postag.startswith('R'):
          return wordnet.ADV
      else:
          return ''
    except:
      print(sys.exc_info()[0])
  
  def __EDA(self):
    try:
      print('EDA by giving word cloud')
      _wordcloud = WordCloud()
      tokens = []
      lst = [[tokens.append(y) for y in x] for x in self.__papersRawData['headline_text_processed']]
      longstring = ','.join(tokens)
      
      _wordcloud.generate(longstring)
      plt.imshow(_wordcloud)
      plt.axis('off')
      plt.show()
      
      self.id2word = gensim.corpora.Dictionary(self.__papersRawData['headline_text_processed'])
      count=0;
      for key, val in self.id2word.iteritems():
            print(key,'-->', val)
            count +=1
            if count >= 10:
                break;
      print('show top 10 words by count')

      self.processeddata=self.__papersRawData['headline_text_processed']
      self.gensim_bow = [self.id2word.doc2bow(text) for text in self.processeddata]
      print('gensim bag of words (wordid : wordcount) \n',self.gensim_bow[:10])
      gensim_tfidf = gensim.models.TfidfModel(self.gensim_bow)
      print('gensim Tfidf data (wordid : Tfidf value) \n')
      self.corpus_tfidf = gensim_tfidf[self.gensim_bow]
      for doc in self.corpus_tfidf[:10]:
            print(doc)
  
    except:
      print(sys.exc_info())
  
  def __printinfo_raw(self):
    print('raw text info')
    print(self.__papersRawData.info())
    print(self.__papersRawData.head())




# In[ ]:




