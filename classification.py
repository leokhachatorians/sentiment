import nltk
import pickle
import random

from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode


documents = []
all_words = []
allowed_word_types = ["J"]

short_pos = open("short_reviews/positive.txt", "r", errors="replace").read()
short_neg = open("short_reviews/negative.txt", "r", errors="replace").read()

for p in short_pos.split('\n'):
    documents.append((p, 'pos'))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p, 'neg'))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save)
save.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]
save = open("pickled_algos/word_features_5k.pickle", "wb")
pickle.dump(word_features, save)
save.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

feature_sets = [(find_features(rev), category) for (rev, category) in documents]
save = open("feature_sets.pickle", "wb")
pickle.dump(feature_sets, save)
save.close()

random.shuffle(feature_sets)

training_set = feature_sets[:10000]
testing_set = feature_sets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Accuracy %: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)
save = open("pickled_algos/original_NB_5k.pickle", "wb")
pickle.dump(classifier, save)
save.close()

mnb_classifier = SklearnClassifier(MultinomialNB())
mnb_classifier.train(training_set)
print("Multinomial Naive Bayes Accuracy %: ", (nltk.classify.accuracy(mnb_classifier, testing_set)) * 100)
save = open("pickled_algos/multi_NB_5k.pickle", "wb")
pickle.dump(mnb_classifier, save)
save.close()

ber_classifier = SklearnClassifier(BernoulliNB())
ber_classifier.train(training_set)
print("Beroulli Naive Bayes Accuracy %: ", (nltk.classify.accuracy(ber_classifier, testing_set)) * 100)
save = open("pickled_algos/ber_NB_5k.pickle", "wb")
pickle.dump(ber_classifier, save)
save.close()

lr_classifier = SklearnClassifier(LogisticRegression())
lr_classifier.train(training_set)
print("Logistic Regression Naive Bayes Accuracy %: ", (nltk.classify.accuracy(lr_classifier, testing_set)) * 100)
save = open("pickled_algos/lr_NB_5k.pickle", "wb")
pickle.dump(lr_classifier, save)
save.close()

sgd_classifier = SklearnClassifier(SGDClassifier())
sgd_classifier.train(training_set)
print("SGD Naive Bayes Accuracy %: ", (nltk.classify.accuracy(sgd_classifier, testing_set)) * 100)
save = open("pickled_algos/sgd_NB_5k.pickle", "wb")
pickle.dump(sgd_classifier, save)
save.close()

lin_svc_classifier = SklearnClassifier(LinearSVC())
lin_svc_classifier.train(training_set)
print("LinearSVC Naive Bayes Accuracy %: ", (nltk.classify.accuracy(lin_svc_classifier, testing_set)) * 100)
save = open("pickled_algos/lin_svc_NB_5k.pickle", "wb")
pickle.dump(lin_svc_classifier, save)
save.close()

nu_svc_classifier = SklearnClassifier(NuSVC())
nu_svc_classifier.train(training_set)
print("NuSVC Naive Bayes Accuracy %: ", (nltk.classify.accuracy(nu_svc_classifier, testing_set)) * 100)
save = open("pickled_algos/nu_svc_NB_5k.pickle", "wb")
pickle.dump(nu_svc_classifier, save)
save.close()
