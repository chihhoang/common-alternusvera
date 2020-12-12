# -*- coding: utf-8 -*-
"""SocialCredibility.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/155pg_WkViHJmDVHIBbedV-a-UCVh_CDc
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import tweepy
import configparser
import json
import pickle


class CerealKillers_SocialCredibility:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('/content/CerealKillers_AlternusVera/SC/twitter_api.ini')
        self.user_info = {}
        self.user_data = []
        self.consumer_key = config.get('default', 'apikey')
        self.consumer_secret = config.get('default', 'apisecretkey')
        self.access_token = config.get('default', 'accesstoken')
        self.access_token_secret = config.get('default', 'accesstokensecret')
        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

    # Call this first when "name" is present
    def search_user_by_name(self, name):
        search = self.api.search_users(name)
        if search:
            self.user_info = search[0]
            self.set_user_data()
        else:
            self.user_data = [0, 0, 0, 0, 0, 0, 0]

    # Call this second to get data for get_prediction()
    # or just call get_prediction(get_user_data())
    def get_user_data(self):
        return self.user_data

    def predict(self, data):
        X = pd.DataFrame([data], columns=['followers', 'favorites', 'friends',
                                          'listed_count', 'statuses_count', 'status_retweeted', 'status_favorited'])
        #RFC = pickle.load( open("/content/CerealKillers_AlternusVera/SC/SocialCredibility.pkl", "rb") )
        # RFC = pickle.load( open("/content/CerealKillers_AlternusVera/SC/SocialCredibilityV2.pkl", "rb") )
        RFC = pickle.load(open("SocialCredibilityV2.pkl", "rb"))
        y = RFC.predict(X)
        return y[0]

    def get_twitter_user_info(self, userid):
        info = {}
        try:
            info = self.api.get_user(int(userid))
        except:
            pass
        finally:
            self.user_info = info
            self.set_user_data()

    def get_user_id(self, info):
        return self.user_info['id']

    def set_user_data(self):
        info = self.user_info
        fol = self.get_followers(info)
        fri = self.get_friends(info)
        fav = self.get_favorites(info)
        lc = self.get_lc(info)
        sc = self.get_sc(info)
        sr = self.get_sr(info)
        sf = self.get_sf(info)
        self.user_data = [fol, fri, fav, lc, sc, sr, sf]

    def get_followers(self, info):
        if bool(info):
            if 'followers_count' in info:
                return info['followers_count']
            else:
                return 0
        else:
            return 0

    def get_friends(self, info):
        if bool(info):
            if 'friends_count' in info:
                return info['friends_count']
            else:
                return 0
        else:
            return 0

    def get_favorites(self, info):
        if bool(info):
            if 'favourites_count' in info:
                return info['favourites_count']
            else:
                return 0
        else:
            return 0

    def get_lc(self, info):
        if bool(info):
            if 'listed_count' in info:
                return info['listed_count']
            else:
                return 0
        else:
            return 0

    def get_sc(self, info):
        if bool(info):
            if 'statuses_count' in info:
                return info['statuses_count']
            else:
                return 0
        else:
            return 0

    def get_sr(self, info):
        if bool(info):
            if 'status' in info:
                if 'retweet_count' in info['status']:
                    return info['status']['retweet_count']
                else:
                    return 0
            else:
                return 0
        else:
            return 0

    def get_sf(self, info):
        if bool(info):
            if 'status' in info:
                if 'favorite_count' in info['status']:
                    return info['status']['favorite_count']
                else:
                    return 0
            else:
                return 0
        else:
            return 0
