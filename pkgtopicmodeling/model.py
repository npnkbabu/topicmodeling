#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
from tqdm import trange
import tqdm
import gensim
import pickle


# In[8]:


class model:
    def __init__(self,data):
        print('model class instantiating')
        self.__data = data
        self.__modelfilename = 'topicmodel.pkl'
    
    def createbasemodel(self):
        print('Creating base model')
        #Topics	Alpha	Beta	Coherence
        #6	asymmetric	symmetric	0.723863804

        self.__model = LdaMulticore(corpus=self.__data.corpus_tfidf, id2word=self.__data.id2word,num_topics=6,
                                    alpha='asymmetric',eta='symmetric',
                                    workers=2,random_state=100,chunksize=100,passes=10,per_word_topics=True)
        print(self.__model.print_topics())
        print(self.__model[self.__data.gensim_bow])
        print('calculating coherence')
        __cohe_model = CoherenceModel(model=self.__model,texts=self.__data.processeddata,dictionary=self.__data.id2word,coherence='c_v')
        __cohe = __cohe_model.get_coherence()
        print('coherence :',__cohe)
        #print('hyper param tuning')
        #self.__hyperparamtunning()
        print('saving model')
        self.__savemodel()
        
        
    def __savemodel(self):
        with open(self.__modelfilename,'wb') as file:
            pickle.dump(self.__model,file)
        
    def __getcoh(self,corpus, dictionary, k, a, b):
        
        __model = LdaMulticore(corpus=corpus, 
                               id2word=dictionary,
                               num_topics=k,
                               alpha=a,
                               eta=b,
                               random_state=100,
                               chunksize=100,
                               passes=10,
                               per_word_topics=True)
        
        __cohe_model = CoherenceModel(model=__model,
                                      texts=self.__data.processeddata,
                                      dictionary=dictionary,
                                      coherence='c_v')
        
        return __cohe_model.get_coherence()
    
    def __hyperparamtunning(self):
        print('hyper param tuning')
        topics_range = list(np.arange(2,10,1))
        alpha_range = list(np.arange(0.01,1,0.3))
        alpha_range.extend(['symmetric','asymmetric'])
        beta_range = list(np.arange(0.01,1,0.3))
        beta_range.extend(['symmetric'])
        
        noofdocs = len(self.__data.processeddata)
        corpus = self.__data.corpus_tfidf
        print('no of docs : ',noofdocs)
        print('dividing corpus  0.25, 0.5, 0.75, 1 shares for testing ')
        corpus_sets = [#gensim.utils.ClippedCorpus(corpus,noofdocs* 0.25),
                      #gensim.utils.ClippedCorpus(corpus,noofdocs* 0.5),
                      #gensim.utils.ClippedCorpus(corpus,noofdocs* 0.75),
                      corpus]
        corpus_title = [ '100% corpus']
        model_results = {'Validation_Set': [],
                         'Topics': [],
                         'Alpha': [],
                         'Beta': [],
                         'Coherence': []
                        }
        if 1==1:
            pbar = tqdm.tqdm(total=540)
            for i in range(len(corpus_sets)):
                for k in topics_range:
                    for a in alpha_range:
                        for b in beta_range:
                            cv = self.__getcoh(corpus_sets[i],self.__data.id2word,k,a,b)
                            model_results['Validation_Set'].append(corpus_title[i])
                            model_results['Topics'].append(k)
                            model_results['Alpha'].append(a)
                            model_results['Beta'].append(b)
                            model_results['Coherence'].append(cv)
                            pbar.update(1)
                            
            results = pd.DataFrame(model_results)
            results.to_csv('lda_tuning_results.csv',index=False)
            print(results)
            pbar.close()
        

    
        
        


# In[ ]:




