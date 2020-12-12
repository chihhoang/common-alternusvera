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


class NetworkBasedPredictor():

  def __init__(self):
    self.model = None

  def __convert2vector(self, tweetToPredict, nlp): 
        textToPredict = str(tweetToPredict)
        review = nlp(textToPredict)
        nlpx_tweet = []
        vector_tweet=0
        for token in review:
          if(token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
              if (len(token.text) > 2 and token.text != 'https' ):
                  vector_tweet += (nlp.vocab[token.text].vector)
        if len(review) != 0:
            vector_tweet = vector_tweet / len(review)
        nlpx_tweet.append(vector_tweet)
        df_test_text = pd.DataFrame(nlpx_tweet) 
        return df_test_text
  
  def __convert_prediction(self, prediction):
    r = [0.16, 0.33, 0.49, 0.66, 0.83, 0.96]
    return r[prediction[0]-1]

  def load(self, path):
    import pickle  # Import pickle Package
    # Load the Model back from file
    with open(path, 'rb') as file:  
        self.model = pickle.load(file)

  def predict(self, text, nlp, source=0):
    df = self.__convert2vector(text, nlp)
    df['node_rank'] = 0
    return self.__convert_prediction(self.model.predict(df))