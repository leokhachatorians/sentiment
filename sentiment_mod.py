import nltk
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
import pickle

from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)

        choice = votes.count(mode(votes))
        conf = choice / len(votes)
        return conf

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

with open("pickled_algos/documents.pickle", "rb") as f:
    documents = pickle.load(f)

with open("pickled_algos/word_features_5k.pickle", "rb") as f:
    word_features = pickle.load(f)

with open("pickled_algos/feature_sets.pickle", "rb") as f:
    feature_sets = pickle.load(f)

#####################
#### Classifiers ####
#####################

with open("pickled_algos/original_NB_5k.pickle", "rb") as f:
    original_NB = pickle.load(f)

with open("pickled_algos/ber_NB_5k.pickle", "rb") as f:
    bernoulli_NB = pickle.load(f)

with open("pickled_algos/lin_svc_NB_5k.pickle", "rb") as f:
    linear_SVC = pickle.load(f)

with open("pickled_algos/lr_NB_5k.pickle", "rb") as f:
    logistic_regression = pickle.load(f)

with open("pickled_algos/multi_NB_5k.pickle", "rb") as f:
    multinomial_NB = pickle.load(f)

with open("pickled_algos/nu_svc_NB_5k.pickle", "rb") as f:
    nu_SVC = pickle.load(f)

with open("pickled_algos/sgd_NB_5k.pickle", "rb") as f:
    sgd = pickle.load(f)

voted_classifier = VoteClassifier(
    original_NB, bernoulli_NB, linear_SVC,
    logistic_regression, multinomial_NB, nu_SVC,
    sgd)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
