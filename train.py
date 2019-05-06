import csv
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot
import scikitplot
import time
import datetime
import statistics
import re


from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_predict

#TODO helpful_votes, total_votes, vine, verified_purchase MAY BE USED AS WEIGHTS
#TODO case, stem, stopwords
#TODO guessed star vs actual difference
#TODO more test files

###############################################################
# BEGIN CONFIGURATION SETUP 
###############################################################
def findClassifier (classifier_name):
    for element in classifiers:
        if element[0] == classifier_name:
            return element
    print("NO VALID CLASSIFIER CHOSEN!")
    quit()

#possible classifiers
classifiers = [
    ("nu_svc" , NuSVC(gamma = 'auto')),
    ("svc" , SVC(gamma = 'auto') ),
    ("svc2" , SVC(kernel="linear", C=0.025)),
    ("decision_tree" , DecisionTreeClassifier(max_depth=5)),
    ("random_forest" , RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ("mlp" , MLPClassifier(alpha=1)),
    ("nb" , MultinomialNB())
]

#scoring metrics
scoring = ['precision_macro', 'recall_macro', 'f1_macro']

#possible csv
vg_set = "Amazon Review Datasets/truncations/vg_trunc_30k.csv"
video_game_data = "Amazon Review Datasets/video_games_truncated.csv"
kitchen_data = "Amazon Review Datasets/kitchen_truncated.csv"

#possible features
features = [
    "review_headline",
    #"review_body"
    #"review_combined" #TODO combine these
]

#possible types of classification
catagory_types = [
    #"boolean",
    "catagories"
]

#possible labels
star_label = "star_rating"

#features being selected
testing_features = ["review_body"]
testing_labels = star_label
selected_dataset = vg_set
selected_classifiers = ["nb"]
testing_types = ["boolean"]

plot_confusion_matrix = True

#use all options
use_all_features = True
use_all_classifiers = False
use_all_types = True

############################################################
# END CONFIGURATION SET UP
############################################################
testing_classifiers = []

for classifier in selected_classifiers:
    testing_classifiers.append(findClassifier(classifier))

if use_all_features:
    testing_features = features

if use_all_classifiers:
    testing_classifiers = classifiers

if use_all_types:
    testing_types = catagory_types

##########################################################
#Generate Dataset
##########################################################

def testDataset(classifier, testing_feature, data_type):
    classifier_name = classifier[0]
    classifier = classifier[1]
    
    if ( data_type == "boolean" ):
        binary_classification = True
    else:
        binary_classification = False

    #input data from CSV file
    input_data = pd.read_csv(selected_dataset)

    start_time = time.time()

    #define dataset
    dataset = {"review_body": input_data["review_body"], "star_rating": input_data["star_rating"], "review_headline": input_data["review_headline"] }
    dataset = pd.DataFrame(data = dataset)
    #print("Dataset size before dropping: " + str(len(dataset)))
    dataset = dataset.dropna() #drop missing values
    #print("Dataset size after dropping: " + str(len(dataset)))


    if binary_classification: # omit 3 star reviews, binary classification
        dataset["label"] = dataset[star_label].apply(lambda rating : +1 if str(rating) > '3' else -1)
    else:
        dataset["label"] = dataset[star_label] # star classification


    body_head_combined = dataset["review_body"].map(str) + dataset["review_headline"].map(str)   
    X = pd.DataFrame(body_head_combined)
    X = pd.DataFrame(dataset, columns = ["review_headline"])

    y = pd.DataFrame(dataset, columns = ["label"])

    #####################################
    # Random Undersampling
    #####################################

    rus = RandomUnderSampler(random_state=13)
    X, y = rus.fit_resample(X, y)
    X = pd.DataFrame(X, columns = [testing_feature])
    #print("\nDataset size after RUS: " + str(len(X)) + "\n")


    #####################################
    # Vectorization / Feature Extraction
    #####################################

    # vectorize the data using a union of features
    union = FeatureUnion([
                ("dots", FunctionFeaturizer(dots)), #works well on headlines, but not text. 
                ("emojis", FunctionFeaturizer(emojis)),
                ("count_exclamation_mark", FunctionFeaturizer(exclamation)),
                ("vectorizer", TfidfVectorizer( token_pattern=r'\b\w+\b', ngram_range=(1,2)))
                  ])
    # fit the above transformers to the data
    X = union.fit_transform(X[testing_feature])


    #####################################
    ## Training with cross validation
    #####################################
    scores = cross_validate(classifier, X, y.ravel(), scoring=scoring, cv=5, return_train_score=False)

    finish_time = time.time()
    elapsed_time = round(finish_time-start_time, 1)

    #Performance Metrics: f1, Precision, Recall
    avgf1 = round(statistics.mean(scores['test_f1_macro']), 3)
    avgprec = round(statistics.mean(scores['test_precision_macro']), 3)
    avgrecall = round(statistics.mean(scores['test_recall_macro']), 3)

    #print(datetime.datetime.now())
    print( "Classifier: " + classifier_name + "\tTesting Feature: " + testing_feature + "\tIs Binary: " + str(binary_classification))
    print( "Precision Score: " + str(avgprec))
    print( "Recall Score: " + str(avgrecall))
    print( "f1 Score: " + str(avgf1))
    print( "Elapsed Time: " + str(elapsed_time) + " seconds")


    if plot_confusion_matrix:
        plotConfusionMatrix(classifier, X, y)

def plotConfusionMatrix(classifier, test_vector, test_y):
    predictions = cross_val_predict(classifier, test_vector, test_y.ravel(), cv = 12)
    scikitplot.metrics.plot_confusion_matrix( test_y.ravel(), predictions, normalize=True)
    matplotlib.pyplot.savefig("save.png") ##add timestamp to title to preserve multiples
    matplotlib.pyplot.show()


#########################################################
# CUSTOM FEATURES
#########################################################

# Feature Functions. Write functions which take the document text
# as a parameter and return a single integer.
# For example: emojis(text) returns the count of ":(" emojis in the text.

def caps(text):
    """Find the longest run of capitol letters and return their length."""
    runs = sorted(re.findall(r"[A-Z]+", text), key=len)
    if runs:
        return len(runs[-1])
    else:
        return 0

def emojis(text):
    sadface = len(re.findall(":\(", text)) + len(re.findall("\):", text))
    happyface = len(re.findall(":\)", text)) + len(re.findall("\(:", text)) + len(re.findall(":D", text))
    if sadface > happyface:
        return 1
    return 0


def exclamation(text):
    return len(re.findall("!", text))

def dots(text):
    return len(re.findall("...", text))
    #This feature meant to detect long pauses.
    # such as: "This thing is ... ok"
    # seems to improve performance on headlines ONLY
    # (tested with NB and nu_svc)

def length(text):
    return len(text)


# The FunctionFeaturizer implements a transformer which can be used in a Feature Union pipeline.
# It allows you to specify the function with which to transform the data, and applies
# the function to each resulting vector in the dataset

class FunctionFeaturizer(TransformerMixin):
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fvs = []
        for datum in X:
            fv = [f(datum) for f in self.featurizers]
            fvs.append(fv)
        return np.array(fvs)


#########################################################
# RUN 
#########################################################

for classifier in testing_classifiers:
    for feature in testing_features:
        for data_type in testing_types:
            testDataset(classifier, feature, data_type)
