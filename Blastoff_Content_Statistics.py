from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from time import time
from os import path
from zipfile import ZipFile
import pandas as pd
import nltk
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google_drive_downloader import GoogleDriveDownloader as gdd
nltk.download('vader_lexicon')
nltk.download('punkt') # word_tokenize
nltk.download('averaged_perceptron_tagger') # pos_tag

import keras
from keras import layers

try:
    import spacy
    sp_lg = spacy.load('en_core_web_lg')
except:
    print('spaCy not installed, install with "python3 -m spacy download en_core_web_lg" then restart the runtime.')

def spacy_large_ner(document):
    return [ent.label_ for ent in sp_lg(document).ents]

def ner_counter(ner_list):
    pers = 0
    norp = 0
    fac = 0
    org = 0
    gpe = 0
    loc = 0
    prod = 0
    eve = 0
    woa = 0
    law = 0
    lang = 0
    dat = 0
    tim = 0
    per = 0
    mon = 0
    quan = 0
    ordi = 0
    cardi = 0

    for n in ner_list:
        if n == 'PERSON':
            pers += 1
        elif n == 'NORP':
            norp += 1
        elif n == 'FAC':
            fac += 1
        elif n == 'ORG':
            org += 1
        elif n == 'GPE':
            gpe += 1
        elif n == 'LOC':
            loc += 1
        elif n == 'PRODUCT':
            prod += 1
        elif n == 'EVENT':
            eve += 1
        elif n == 'WORK_OF_ART':
            woa += 1
        elif n == 'LAW':
            law += 1
        elif n == 'LANGUAGE':
            lang += 1
        elif n == 'DATE':
            dat += 1
        elif n == 'TIME':
            tim += 1
        elif n == 'PERCENT':
            per += 1
        elif n == 'MONEY':
            mon += 1
        elif n == 'QUANTITY':
            quan += 1
        elif n == 'ORDINAL':
            ordi += 1
        elif n == 'CARDINAL':
            cardi += 1
    return [pers, norp, fac, org, gpe, loc, prod, eve, woa, law, lang, dat, tim,\
            per, mon, quan, ordi, cardi]

class BlastoffContentStatistics(object):

    def __init__(self, dpos=True, sa=True, ner=True, mpath=None):
        self.dpos = dpos
        self.sa = sa
        self.ner = ner
        self.clf = None
        self.encoder = None

        gdd.download_file_from_google_drive(file_id='1oTfsNgkmEBemkVfrWSjnc-K9svmL7Iak',
                                            dest_path='./bcs_encoder.zip',
                                            unzip=False)

        archive = ZipFile('bcs_encoder.zip')
        for file in archive.namelist():
            archive.extract(file, './')
        self.encoder = keras.models.load_model('./bcs_encoder')

        if mpath:
            self.mpath = mpath
            if path.exists(self.mpath):
                self.clf = pickle.load(open(self.mpath, 'rb'))
            else:
                print("Model not found: " + self.mpath)
                try:
                    print("Loading default pretrained model.")
                    gdd.download_file_from_google_drive(file_id='1oFjoL9LWrp2-YPSJL2UhBQ1efV9LiC2n',
                                                        dest_path='./content_statistic_model.pickle',
                                                        unzip=False)
                    self.clf = pickle.load(open('content_statistic_model.pickle', 'rb'))
                except:
                    print("Unable to load default model. Please contact author or train a new one.")
        else:
            try:
                print("Loading default pretrained model.")
                gdd.download_file_from_google_drive(file_id='1oFjoL9LWrp2-YPSJL2UhBQ1efV9LiC2n',
                                                    dest_path='./content_statistic_model.pickle',
                                                    unzip=False)
                self.clf = pickle.load(open('content_statistic_model.pickle', 'rb'))
            except:
                print("Unable to load default model. Please contact author or train a new one.")

    def extract(self, _X):
        if 'Statement' in _X.columns:
            outdf = _X.loc[:, ['Statement']]
            postags = []
            for i, c in enumerate(_X['Statement']):
                postags.append(nltk.pos_tag(nltk.word_tokenize(c)))
            
            tag_counters = []
            for tags in postags:
                tag_counter = {
                    '$': 0,
                    "''": 0,
                    '(': 0,
                    ')': 0,
                    ',': 0,
                    '--': 0,
                    '.': 0,
                    ':': 0,
                    'CC': 0,
                    'CD': 0,
                    'DT': 0,
                    'EX': 0,
                    'FW': 0,
                    'IN': 0,
                    'JJ': 0,
                    'JJR': 0,
                    'JJS': 0,
                    'LS': 0,
                    'MD': 0,
                    'NN': 0,
                    'NNP': 0,
                    'NNPS': 0,
                    'NNS': 0,
                    'PDT': 0,
                    'POS': 0,
                    'PRP': 0,
                    'PRP$': 0,
                    'RB': 0,
                    'RBR': 0,
                    'RBS': 0,
                    'RP': 0,
                    'SYM': 0,
                    'TO': 0,
                    'UH': 0,
                    'VB': 0,
                    'VBD': 0,
                    'VBG': 0,
                    'VBN': 0,
                    'VBP': 0,
                    'VBZ': 0,
                    'WDT': 0,
                    'WP': 0,
                    'WP$': 0,
                    'WRB': 0,
                    '``': 0,
                    '#': 0
                }
                for tag in tags:
                    if tag[1] in tag_counter:
                        tag_counter[tag[1]] += 1
                    else:
                        print(tag)
                tag_counters.append(tag_counter)

            if self.dpos:
                ccl = []
                cdl = []
                dtl = []
                exl = []
                fwl = []
                inl = []
                jjl = []
                lsl = []
                mdl = []
                nnl = []
                pdtl = []
                posl = []
                prpl = []
                rbl = []
                rpl = []
                syml = []
                tol = []
                uhl = []
                vbl = []
                whl = []

                for tct in tag_counters:
                    tct_values = list(tct.values())
                    ccl.append(tct_values[8])
                    cdl.append(tct_values[9])
                    dtl.append(tct_values[10])
                    exl.append(tct_values[11])
                    fwl.append(tct_values[12])
                    inl.append(tct_values[13])
                    jjl.append(tct_values[14]+tct_values[15]+tct_values[16])
                    lsl.append(tct_values[17])
                    mdl.append(tct_values[18])
                    nnl.append(tct_values[19]+tct_values[20]+tct_values[21]+tct_values[22])
                    pdtl.append(tct_values[23])
                    posl.append(tct_values[24])
                    prpl.append(tct_values[25]+tct_values[26])
                    rbl.append(tct_values[27]+tct_values[28]+tct_values[29])
                    rpl.append(tct_values[30])
                    syml.append(tct_values[0]+tct_values[1]+tct_values[2]+tct_values[3]+
                                tct_values[4]+tct_values[5]+tct_values[6]+tct_values[7]+
                                tct_values[31]+tct_values[44]+tct_values[45])
                    tol.append(tct_values[32])
                    uhl.append(tct_values[33])
                    vbl.append(tct_values[34]+tct_values[35]+tct_values[36]+tct_values[37]+
                               tct_values[38]+tct_values[39])
                    whl.append(tct_values[40]+tct_values[41]+tct_values[42]+tct_values[43])

                outdf['CC'] = ccl
                outdf['CD'] = cdl
                outdf['DT'] = dtl
                outdf['EX'] = exl
                outdf['FW'] = fwl
                outdf['IN'] = inl
                outdf['JJ'] = jjl
                outdf['LS'] = lsl
                outdf['MD'] = mdl
                outdf['NN'] = nnl
                outdf['PDT'] = pdtl
                outdf['POS'] = posl
                outdf['PRP'] = prpl
                outdf['RB'] = rbl
                outdf['RP'] = rpl
                outdf['SYM'] = syml
                outdf['TO'] = tol
                outdf['UH'] = uhl
                outdf['VB'] = vbl
                outdf['WH'] = whl
            else:
                jjl = []
                nnl = []
                rbl = []
                vbl = []

                for tct in tag_counters:
                    tct_values = list(tct.values())
                    jjl.append(tct_values[14]+tct_values[15]+tct_values[16])
                    nnl.append(tct_values[19]+tct_values[20]+tct_values[21]+tct_values[22])
                    rbl.append(tct_values[27]+tct_values[28]+tct_values[29])
                    vbl.append(tct_values[34]+tct_values[35]+tct_values[36]+tct_values[37]+
                               tct_values[38]+tct_values[39])
                
                outdf['JJ'] = jjl
                outdf['NN'] = nnl
                outdf['RB'] = rbl
                outdf['VB'] = vbl

            if self.sa:
                sid = SentimentIntensityAnalyzer()
                SA_C = []
                SA_P = []
                SA_NU = []
                SA_NG = []

                for sentence in _X['Statement']:
                    ss = sid.polarity_scores(sentence)
                    for k in sorted(ss):
                        if k == 'compound':
                            SA_C.append(ss[k])
                        elif k == 'pos':
                            SA_P.append(ss[k])
                        elif k == 'neu':
                            SA_NU.append(ss[k])
                        elif k == 'neg':
                            SA_NG.append(ss[k])
                outdf['SA_C'] = SA_C
                outdf['SA_P'] = SA_P
                outdf['SA_NU'] = SA_NU
                outdf['SA_NG'] = SA_NG

            if self.ner:
                NER_pers = []
                NER_norp = []
                NER_fac = []
                NER_org = []
                NER_gpe = []
                NER_loc = []
                NER_prod = []
                NER_eve = []
                NER_woa = []
                NER_law = []
                NER_lang = []
                NER_dat = []
                NER_tim = []
                NER_per = []
                NER_mon = []
                NER_quan = []
                NER_ordi = []
                NER_cardi = []

                for statement in _X['Statement']:
                    ner_count = ner_counter(spacy_large_ner(statement))
                    NER_pers.append(ner_count[0])
                    NER_norp.append(ner_count[1])
                    NER_fac.append(ner_count[2])
                    NER_org.append(ner_count[3])
                    NER_gpe.append(ner_count[4])
                    NER_loc.append(ner_count[5])
                    NER_prod.append(ner_count[6])
                    NER_eve.append(ner_count[7])
                    NER_woa.append(ner_count[8])
                    NER_law.append(ner_count[9])
                    NER_lang.append(ner_count[10])
                    NER_dat.append(ner_count[11])
                    NER_tim.append(ner_count[12])
                    NER_per.append(ner_count[13])
                    NER_mon.append(ner_count[14])
                    NER_quan.append(ner_count[15])
                    NER_ordi.append(ner_count[16])
                    NER_cardi.append(ner_count[17])

                outdf['NER_pers'] = NER_pers
                outdf['NER_norp'] = NER_norp
                outdf['NER_fac'] = NER_fac
                outdf['NER_org'] = NER_org
                outdf['NER_gpe'] = NER_gpe
                outdf['NER_loc'] = NER_loc
                outdf['NER_prod'] = NER_prod
                outdf['NER_eve'] = NER_eve
                outdf['NER_woa'] = NER_woa
                outdf['NER_law'] = NER_law
                outdf['NER_lang'] = NER_lang
                outdf['NER_dat'] = NER_dat
                outdf['NER_tim'] = NER_tim
                outdf['NER_per'] = NER_per
                outdf['NER_mon'] = NER_mon
                outdf['NER_quan'] = NER_quan
                outdf['NER_ordi'] = NER_ordi
                outdf['NER_cardi'] = NER_cardi

            return outdf.drop(['Statement'], axis=1)
        else:
            print('Statement column not found.')
            return

    def fit(self, _X_train, _y_train, _X_test, _y_test):
        if 'Statement' in _X_train.columns and 'Statement' in _X_test.columns:
            X_train = _X_train.copy()
            X_test = _X_test.copy()
        elif 'Statement' in _X_train.columns:
            print('Statement column not found in the testing data.')
            return
        elif 'Statement' in _X_test.columns:
            print('Statement column not found in the training data.')
            return

        X_train = self.extract(X_train)
        X_test = self.extract(X_test)

        y_train = _y_train.replace('pants-fire', 0).replace('false', 1).replace('barely-true', 2).replace('half-true', 3).replace('mostly-true', 4).replace('true', 5)
        y_test = _y_test.replace('pants-fire', 0).replace('false', 1).replace('barely-true', 2).replace('half-true', 3).replace('mostly-true', 4).replace('true', 5)

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes", "QDA"]

        classifiers = [
            KNeighborsClassifier(2),
            SVC(kernel="linear", C=0.025, probability=True, random_state=0),
            SVC(gamma=2, C=1, probability=True, random_state=0),
            DecisionTreeClassifier(max_depth=5, random_state=0),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=0),
            MLPClassifier(alpha=1, max_iter=1000, random_state=0),
            AdaBoostClassifier(random_state=0),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

        max_score = 0.0
        max_class = ''
        for name, clf in zip(names, classifiers):
            start_time = time()
            clf.fit(X_train, y_train)
            score = 100.0 * clf.score(X_test, y_test)
            print('Classifier = %s, Score (test, accuracy) = %.2f,' %(name, score), 'Training time = %.2f seconds' % (time() - start_time))
            
            if score > max_score:
                self.clf = clf
                max_score = score
                max_class = name

        print(80*'-' )
        print('Best --> Classifier = %s, Score (test, accuracy) = %.2f' %(max_class, max_score))


    def predict(self, _X):
        if self.clf:
            if 'Statement' in _X.columns:
                X = _X.copy()
                X = self.extract(X)
                return pd.DataFrame(data=self.clf.predict(X), columns=['Label']).replace(0, 'pants-fire').replace(1, 'false').replace(2, 'barely-true').replace(3, 'half-true').replace(4, 'mostly-true').replace(5, 'true')
            else:
                print('Statement column not found.')
                return
        else:
            print('Model not found.')


    def export(self, mpath):
        if path.isdir(mpath):
            pickle.dump(self.clf, open(path.join(mpath, 'content_statistic_model.pickle'), 'wb'))
        else:
            print("Path not available.")


    def encode(self, _X):
        if self.encoder:
            if 'Statement' in _X.columns:
                X = self.extract(_X)
                return self.encoder.predict(X)
            else:
                print('Statement column not found.')
                return
        else:
            print('Encoder not found.')
            return