from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
import pandas as pd
import pickle


global stops
global stemmer
global wlemmatizer
global tokenizer

stops = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.SnowballStemmer('english')
wlemmatizer = WordNetLemmatizer()
tokenizer = nltk.word_tokenize

def cleaning(raw_news):
    
    import nltk
    global stops
    global stemmer
    global wlemmatizer
    
    # 1. Remove non-letters/Special Characters and Punctuations
    news = re.sub("[^a-zA-Z]", " ", raw_news)
    
    # 2. Convert to lower case.
    news =  news.lower()
    
    # 3. Remove punctuation
    news = news.translate(str.maketrans('', '', string.punctuation))
        
    # 4. Tokenize.
    news_words = tokenizer(news)
        
    # 5. Remove stop words. 
    words = [w for w in news_words if not w in stops]
    
    # 6. Lemmentize 
    wordnet_lem = [wlemmatizer.lemmatize(w) for w in words ]
    
    # 7. Stemming
    stems = [stemmer.stem(w) for w in wordnet_lem ]

    # 8. Join the stemmed words back into one string after removing small words
    return ' '.join(word for word in stems if len(word)>2)


def predictLable(text):
  with open('./event_coverage.pkl', 'rb') as file:  
    nb_pipeline_ngram = pickle.load(file)
  data = {'clean':  [text]
        }
  df_test = pd.DataFrame (data, columns = ['clean'])
  df_test['clean']=text
  df_test['clean'] = df_test['clean'].apply(cleaning) 
 
  cleantxt = df_test['clean'][0]
  predicted = nb_pipeline_ngram.predict([cleantxt])
  predicedProb = nb_pipeline_ngram.predict_proba([cleantxt])[:,1]
  return predicedProb;
