# -*- coding: utf-8 -*-
"""AlternusVera_Topic_LDA_Sprint4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gxwoiBlbXvC4Qehnx07hlIG_ju37H_dh
"""

class Topics_with_LDA_Bigram: 
  def encodeLabel(df):
    df.Label[df.Label == 'FALSE'] = 0
    df.Label[df.Label == 'half-true'] = 1
    df.Label[df.Label == 'mostly-true'] = 1
    df.Label[df.Label == 'TRUE'] = 1
    df.Label[df.Label == 'barely-true'] = 0
    df.Label[df.Label == 'pants-fire'] = 0
    return df

  def sent_to_words(sentences):
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

  def data_preprocess(df):
    import re
    dt_processed = df.News.values.tolist()
    # Remove new line characters
    dt_processed = [re.sub(r'[^\w\s]','',sent) for sent in dt_processed]
    dt_processed = [re.sub("'"," ",sent) for sent in dt_processed]
    data_words_processed = list(Topics_with_LDA_Bigram.sent_to_words(dt_processed))
    return data_words_processed
  
    # Define functions for stopwords, bigrams, trigrams and lemmatization
  def remove_stopwords(texts):
    from nltk.corpus import stopwords
    from gensim.utils import simple_preprocess
    import nltk
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

  def lemmatization(texts):
    # spacy for lemmatization
    import spacy
    """https://spacy.io/api/annotation"""
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc])
    return texts_out

  def extract_bigrams(data):
    from nltk.util import ngrams
    n_grams = ngrams(data, 2)
    return ['_'.join(grams) for grams in n_grams]    

  def create_bigrams():
    Bigrams = []
    for i in range(len(data_words_processed)):
      Bigrams.append(Topics_with_LDA_Bigram.extract_bigrams(data_words_processed[i]))
    return Bigrams    

  def lda_model_final(corpus, id2word):
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel

    lda_model_bigram_scrapped = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    return lda_model_bigram_scrapped  

  def format_topics_sentences(ldamodel, corpus, texts):
    import pandas as pd
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)

  #Calculate sentiment polarity and find the max value. Normalize the encoded label values.
  def sentiment_analyzer_scores(df):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import math
    analyser = SentimentIntensityAnalyzer()
    sentiment_score = []
    sentiment_labels = {0:'negative', 1:'positive', 2:'neutral'}
    for index,row in df.iterrows():
        score = analyser.polarity_scores(row['News'])
        values = [score['neg'], score['pos'], score['neu']]
        max_index = values.index(max(values))
        data = {'sentiment_score':score, 'sentiment_label':sentiment_labels[max_index], 'sentiment_label_encode': 1+math.log(max_index+1)}
        sentiment_score.append(data)
    return sentiment_score 

  #Scalar Vector for testing dataset
  def testing_dataset_vector(df_testing):
    vec_testing = []
    for i in range(len(df_testing['sentiment_encode'])):
        vec = df_testing['sentiment_encode'].iloc[i] + df_testing['Topic_Score'].iloc[i]
        vec_testing.append(vec)
    return vec_testing      

  def getTopicScoreBigramLDAModel(headline):
    import pandas as pd
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel
    from sklearn.linear_model import  LogisticRegression
    #load the model
    import pickle 
    import numpy as np

    #mounting google drive
    from google.colab import drive
    drive.mount('/content/drive/')

    cols = [[headline]]
    df_testing = pd.DataFrame(cols,columns=['News'])
    df_testing.head()

    #encoding the label from text to numeric value
    #df_testing = Topics_with_LDA_Bigram.encodeLabel(df_testing)

    #data pre-process
    dt_words_processed =  Topics_with_LDA_Bigram.data_preprocess(df_testing)

    # Remove Stop Words
    dt_nostops = Topics_with_LDA_Bigram.remove_stopwords(dt_words_processed)

    # Form Bigrams
    dt_words_bigrams = []
    for i in range(len(dt_nostops)):
      dt_words_bigrams.append(Topics_with_LDA_Bigram.extract_bigrams(dt_nostops[i]))  
        
    # Do lemmatization keeping only noun, adj, vb, adv
    dt_lemmatized = Topics_with_LDA_Bigram.lemmatization(dt_words_bigrams)
    print(dt_lemmatized) 


    # Create Dictionary
    id2word = corpora.Dictionary(dt_lemmatized)

    # Create Corpus
    texts = dt_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

      
    #Bigram LDA Model
    lda_model_bigram =  Topics_with_LDA_Bigram.lda_model_final(corpus, id2word)

    #
    dt_topic_sents_keywords = Topics_with_LDA_Bigram.format_topics_sentences(ldamodel=lda_model_bigram, corpus=corpus, texts=dt_words_processed)

    # Format
    dt_dominant_topic = dt_topic_sents_keywords.reset_index()
    dt_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Score', 'Keywords', 'Text']

    # Show
    dt_dominant_topic.head()

    #Distillation - Sentiment analysis score
    sentiment_score_dt = Topics_with_LDA_Bigram.sentiment_analyzer_scores(df_testing)
    print(sentiment_score_dt)

    #append dataset with sentiment label and normalized encoded value
    sentiment_score = pd.DataFrame(sentiment_score_dt)
    df_testing['sentiment'] = sentiment_score['sentiment_label']
    df_testing['sentiment_encode'] = sentiment_score['sentiment_label_encode']
    df_testing['Keywords'] = dt_dominant_topic['Keywords']
    df_testing['Dominant_Topic'] = dt_dominant_topic['Dominant_Topic']
    df_testing['Topic_Score'] = dt_dominant_topic['Topic_Score']
    print(df_testing.head())

    #Get test data for classification

    X_test = np.array(Topics_with_LDA_Bigram.testing_dataset_vector(df_testing))
    X_test =  X_test.reshape(-1, 1)

      
    fake_news_classifier = pickle.load(open('/content/drive/MyDrive/MLFall2020/girlswhocode/models/LogisticRegression_model.sav', 'rb'))
    predicted = fake_news_classifier.predict(X_test)
    predicedProb = fake_news_classifier.predict_proba(X_test)[:,1]

    score = 1 - float(predicedProb)
    return score