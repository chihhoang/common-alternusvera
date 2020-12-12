#!/usr/bin/env python
"""Define a class to predict a Malicous Account."""
import pandas as pd
import numpy as np
import math
import warnings
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from tqdm import tqdm
from io import BytesIO
import requests
from google.colab import drive
import sys
import re
from keras.layers import Embedding, Dense, LSTM, Dense, Input, concatenate
from keras.models import Model
from keras.utils import plot_model
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import xgboost as xgb
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import plot_confusion_matrix
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier.rocauc import roc_auc
import pickle  # Import pickle Package
import nltk
from nltk.stem import PorterStemmer #for stemming
from nltk.tokenize import word_tokenize
import nltk.corpus
nltk.download('treebank')
nltk.download('universal_tagset') 
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import  LogisticRegression
from sklearn import metrics
from sklearn import svm
import requests
from io import BytesIO



class MalicousAccount( object ):

    def __init__( self ):
        ma = 0

    def predict( self, cls, tweetText, nlp ): 
        tweetVector = self.__convert2vector(tweetText, nlp)
        predictionResult =  cls.predict(tweetVector)
        botScoreResult, labelResult = self.__predictionScore(predictionResult) 
        # print('label: ', predictionResult)
        # print('label: ', labelResult)
        # print('score: ', botScoreResult)
        return predictionResult[0], botScoreResult, labelResult
   
   
    def load( self, model2load ):
        import pickle  # Import pickle Packag
        # Load the Model back from file
        with open(model2load, 'rb') as file:  
            Pickled_Model = pickle.load(file)
        return Pickled_Model


    def save(self, *args, **kwargs):
        import pickle  # Import pickle Packag
        from sklearn.ensemble import RandomForestClassifier
        modelName = kwargs.get('modelName', 'MalicousAccount.pkl')
        model2save = kwargs.get('model2save', MLPClassifier(alpha=1, max_iter=1000))
        with open(modelName, 'wb') as file:  
            pickle.dump(model2save, file)
        msg = "saved model " + modelName 
        return msg
    

#===Private funtions
    # Convert to verctor using word2Vec
    def __convert2vector(self, tweetToPredict, nlp): 
        textToPredict = str(tweetToPredict)
        review = nlp(textToPredict)
        nlpx_tweet = []
        vector_tweet=0
        n_tokens=0
        for token in review:
          if(token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
              if (len(token.text) > 2 and token.text != 'https' ):
                  # print(token.text, " ---> ", token.pos_)
                  vector_tweet += (nlp.vocab[token.text].vector)
                  # print(vector_tweet) 
                  n_tokens += 1 
        if n_tokens != 0:
            vector_tweet = vector_tweet / n_tokens
        nlpx_tweet.append(vector_tweet)
        df_test_text = pd.DataFrame(nlpx_tweet) 
        return df_test_text


    def __predictionScore(self, prediction):
        # n_classes = ['Human', 'cyborg', 'botWallE', 'botT800' ]
        if prediction == 1: 
          label = 'Human'
          botScore = 0.90 #(1-0.8)/2
        elif prediction == 2:
          label = 'cyborg'
          botScore = 0.70 #(0.8-0.6)/2
        elif prediction == 3: 
          label = 'botWallE'
          botScore = 0.50 #(0.6-0.4)/2
        elif prediction == 4: 
          label = 'botT800'      
          botScore = 0.30 #(0.4-0.2)/2
        else: 
          label = 'Allient'      
          botScore = 0.10 #(0.2-0.0)/2      
        return botScore, label


    def __text_to_Sentence_clean(self, text):
      ''' 
      Pre process and convert texts to a list of words 
      input: str
      output: list of cleaned word
      '''
      text = str(text)
      text = text.lower()

      # Clean the text
      text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
      text = re.sub(r"what's", "what is ", text)
      text = re.sub(r"rt", " ", text)
      text = re.sub(r"\'s", " ", text)
      text = re.sub(r"\'ve", " have ", text)
      text = re.sub(r"can't", "cannot ", text)
      text = re.sub(r"n't", " not ", text)
      text = re.sub(r"i'm", "i am ", text)
      text = re.sub(r"\'re", " are ", text)
      text = re.sub(r"\'d", " would ", text)
      text = re.sub(r"\'ll", " will ", text)
      text = re.sub(r",", " ", text)
      text = re.sub(r"\.", " ", text)
      text = re.sub(r"!", " ! ", text)
      text = re.sub(r"\/", " ", text)
      text = re.sub(r"\^", " ^ ", text)
      text = re.sub(r"\+", " + ", text)
      text = re.sub(r"\-", " - ", text)
      text = re.sub(r"\=", " = ", text)
      text = re.sub(r"'", " ", text)
      text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
      text = re.sub(r":", " : ", text)
      text = re.sub(r" e g ", " eg ", text)
      text = re.sub(r" b g ", " bg ", text)
      text = re.sub(r" u s ", " american ", text)
      text = re.sub(r"\0s", "0", text)
      text = re.sub(r" 9 11 ", "911", text)
      text = re.sub(r"e - mail", "email", text)
      text = re.sub(r"j k", "jk", text)
      text = re.sub(r"\s{2,}", " ", text)
      text = re.sub("quikly","quickly", text)

      listText = text.split()
      return text, listText


    def __topMostUsedSentence(self, bagofWords, nSentence):
      # bagofWordsArr = np.array(bagofWords)
      df_corpus= pd.DataFrame(bagofWords)
      df_bag = pd.DataFrame(df_corpus.pivot_table(index=['word'], aggfunc='size'))
      df_bagofwords= pd.DataFrame()
      df_bagofwords = df_bag.sort_values(by=0, ascending=False)
      df_bagofwords.reset_index(inplace=True)
      df_bagofwords.rename(columns = {'index':'word'})
      df_bagofwords = df_bagofwords.rename(columns={"word": "word", 0: "quantity"})
      return df_bagofwords[:-nSentence]      


    def __selectEspecificSentence(self, bagFullofnSentence):
      selSentence = []
      n_tokens=0
      review = nlp(bagFullofnSentence)
      for token in review.noun_chunks:
          # if(token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
              if (len(token.text) > 10 and token.text != 'https' ):
                # selSentence.append(token.text)
                selSentence.append({'word': token.text, 'dep_': token.root.dep_})
                # selSentence.append({'text': token.text, 'dep': token.dep_})
              else:
                n_tokens += 1  
      return selSentence, n_tokens 


    def __goldenVector(self, df_evalData): 
        #bot=1(Human)   bot=2(cyborg)   bot=2(botWallE)   bot=4(botT800)
        n_classes = ['Human', 'cyborg', 'botWallE', 'botT800' ]
        corpusNgrams = {}
        df_corpusNgrams = {}
        nlpx = []
        vectorNgrams_golden= {}
        nTopSentence = 10
        for ratingClass in tqdm(range(len(n_classes))):
        # for ratingClass in (range(len(n_classes))):  
            tweetListUnf = df_evalData.query('rating == ' + str(ratingClass + 1)).full_text.to_string(index=False)
            tweetsList, _ = self.__text_to_Sentence_clean(tweetListUnf)
            selectedSentence, _ = self.__selectEspecificSentence(tweetsList)
            corpusNgramsTopList = self.__topMostUsedSentence(selectedSentence, nTopSentence)
            #print(corpusNgramsTopList)
            
            df_corpusNgrams[n_classes[ratingClass]] = corpusNgramsTopList.copy()
            # corpus[n_classes[ratingClass]] = corpusNgramsTopList.word.to_string(index=False)
            vectorNgrams_golden[n_classes[ratingClass]] = 0  
            n_tokens=0  
            for token in corpusNgramsTopList.word.to_string(index=False):
              vectorNgrams_golden[n_classes[ratingClass]] += (nlp.vocab[token].vector)
              n_tokens += 1  
            vectorNgrams_golden[n_classes[ratingClass]] = vectorNgrams_golden[n_classes[ratingClass]] / n_tokens
        return vectorNgrams_golden, df_corpusNgrams


    def __cosine(self, v1, v2):
        vScalar=  np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))  
        return  vScalar  


    def __applySimilarity(self, df_evalData, vectorNgrams_golden): 
        #bot=1(Human)   bot=2(cyborg)   bot=2(botWallE)   bot=4(botT800)
        n_classes = ['Human', 'cyborg', 'botWallE', 'botT800' ]
        corpusNgrams_tweet = {}
        df_corpusNgrams_tweet = {}
        nlpxNgrams_tweet = []
#??????      
        a=0
        for tweetUnf in tqdm(df_evalData.full_text): 
            if (a == 10):
              break
            a += 1  

            tweet, _ = self.__text_to_Sentence_clean(tweetUnf)
            # print('\n', tweet)
            review = nlp(tweet)
            vector_tweet=0
            n_tokens=0
            for token in review.noun_chunks:
                # if (n_tokens == 10):
                #   break
                # print(token.text, " ---> ", token.root.dep_)
                #print(token.text) #, token.root.text, token.root.dep_, token.root.head.text)
                vector_tweet += (nlp.vocab[token.text].vector)
                n_tokens += 1           

            if n_tokens != 0:
                vector_tweet = vector_tweet / n_tokens
                cosine_result = {}
                for ratingClass in range(len(n_classes)):
                    #print('Similarity between tweet and golden '+ n_classes[ratingClass] , self.__cosine(vectorNgrams_golden[n_classes[ratingClass]], vector_tweet))
                    cosine_result[ratingClass] = self.__cosine(vectorNgrams_golden[n_classes[ratingClass]], vector_tweet)
                nlpxNgrams_tweet.append({'tweet_similarity_'+ n_classes[0] : cosine_result[0],
                                  'tweet_similarity_'+ n_classes[1] : cosine_result[1],
                                  'tweet_similarity_'+ n_classes[2] : cosine_result[2],
                                  'tweet_similarity_'+ n_classes[3] : cosine_result[3]})
            else:
                nlpxNgrams_tweet.append({'tweet_similarity_'+ n_classes[0] : 0,
                                  'tweet_similarity_'+ n_classes[1] : 0,
                                  'tweet_similarity_'+ n_classes[2] : 0,
                                  'tweet_similarity_'+ n_classes[3] : 0})

        df_corpusNgrams_tweet =  pd.DataFrame(nlpxNgrams_tweet)
        return df_corpusNgrams_tweet
      

    def __mullerLoop(self, df_evalData_Ngram):
    # def __mullerLoop(self, df_evalData_Ngram, features):
        training_data = df_evalData_Ngram.copy()
        Xng = training_data.iloc[:,:-1]
        yng = training_data.iloc[:,-1:]
        print("X Columns: ")
        print(Xng.columns)
        print("y Columns: ")
        print(yng.columns)
        names = ["Nearest Neighbors", #"Linear SVM", #"RBF SVM", "Gaussian Process",
                "Decision Tree", "Random Forest", "AdaBoost", #"Neural Net",
                "Naive Bayes", "QDA", "XGBoost", "Logistic Reg"]
        classifiers = [
            KNeighborsClassifier(n_neighbors = 5),
            #SVC(kernel="linear", C=0.025),
            #SVC(gamma=2, C=1),
            #GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(criterion='gini', min_samples_leaf=50, min_samples_split=10),
            AdaBoostClassifier(),
            #MLPClassifier(alpha=1, max_iter=1000),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            xgb.XGBClassifier(objective='reg:logistic'),
            LogisticRegression(penalty='l2', C=1.0, solver='liblinear')]

        Xng_train, Xng_test, yng_train, yng_test = \
            train_test_split(Xng, yng.iloc[:,-1], test_size= .30 )   #random_state = 42)  

        rowsLink = []
        max_score = 0.0
        max_class = ''
        # iterate over classifiers
        warnings.filterwarnings('ignore')
        for name, clf in zip(names, classifiers):
            start_time = time()
            ml_cls= clf.fit(Xng_train, yng_train)
            score = 100.0 * clf.score(Xng_test, yng_test)
            
            pred_train = ml_cls.predict(Xng_train)
            # train_accuracy = 100.0 * accuracy_score(yng_train, pred_train)
            train_accuracy = accuracy_score(yng_train, pred_train)
            pred_test = ml_cls.predict(Xng_test)    
            # test_accuracy = 100.0 * accuracy_score(yng_test, pred_test)
            test_accuracy = accuracy_score(yng_test, pred_test)
          
            rowsLink.append([score, train_accuracy, test_accuracy, name, clf, (time() - start_time)])
            print('Classifier = %s, Score (test, accuracy) = %.2f,' %(name, score), 'Train accuracy = %.2f ' % train_accuracy, 
                  'Test accuracy = %.2f ' % test_accuracy, 'Training time = %.2f ' % (time() - start_time))
            if score > max_score:
                clf_best = clf
                max_score = score
                max_class = name
        warnings.resetwarnings()
        print(80*'-' )
        print('Best --> Classifier = %s, Score (test, accuracy) = %.2f' %(max_class, max_score))
        dfCla_mullerloop = pd.DataFrame(rowsLink, columns=["score", "train accuracy", "test accuracy", "name", "classifier", "Time-elapsed"]).sort_values('score', ascending=False).reset_index(drop=True)
        return dfCla_mullerloop