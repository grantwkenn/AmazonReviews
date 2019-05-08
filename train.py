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

from scipy.sparse import coo_matrix, hstack
import configuration

#TODO helpful_votes, total_votes, vine, verified_purchase MAY BE USED AS WEIGHTS
#TODO case, stem, stopwords
#TODO guessed star vs actual difference
#TODO more test files


##########################################################
#Generate Dataset
##########################################################

def testDataset(classifier, testingFeatures, dataType, selectedDataset, scoringMetrics, isPlottingConfusionMatrix):
    classifier_name = classifier[0]
    classifier = classifier[1]
    
    if ( dataType == "boolean" ):
        binary_classification = True
    else:
        binary_classification = False

    #input data from CSV file
    input_data = pd.read_csv(selectedDataset)

    start_time = time.time()

    #define dataset
    dataset = {"review_body": input_data["review_body"].values, 
        "star_rating": input_data["star_rating"].values,
        "review_headline": input_data["review_headline"].values,
        "helpful_votes": input_data["helpful_votes"].values,
        "total_votes": input_data["total_votes"].values
        }
    dataset = pd.DataFrame(data = dataset)
    #print("Dataset size before dropping: " + str(len(dataset)))
    dataset = dataset.dropna() #drop missing values
    #print("Dataset size after dropping: " + str(len(dataset)))


    if binary_classification: # omit 3 star reviews, binary classification
        dataset["label"] = dataset["star_rating"].apply(lambda rating : +1 if str(rating) > '3' else -1)
    else:
        dataset["label"] = dataset["star_rating"] # star classification


    body_head_combined = dataset["review_body"].map(str) + dataset["review_headline"].map(str)   
    #X = pd.DataFrame(body_head_combined)
    #X = pd.DataFrame(dataset, columns = [testingFeatures])
    tableFields = ["review_headline", "review_body", "star_rating", "helpful_votes", "total_votes"]
    X = pd.DataFrame(dataset, columns = tableFields)
    y = pd.DataFrame(dataset, columns = ["label"])

    #####################################
    # Random Undersampling
    #####################################

    rus = RandomUnderSampler(random_state=13)
    X, y = rus.fit_resample(X, y)
    X = pd.DataFrame(X, columns = tableFields)
    #print(X)
    X_headline = X["review_headline"]#pd.DataFrame(X, columns = ["review_headline"])
    #print(X_headline)
    #print("\nDataset size after RUS: " + str(len(X)) + "\n")


    #####################################
    # Vectorization / Feature Extraction
    #####################################

    # vectorize the data using a union of features
    union = FeatureUnion([
                #("dots", FunctionFeaturizer(dots)), #works well on headlines, but not text. 
                #("emojis", FunctionFeaturizer(emojis)),
                #("count_exclamation_mark", FunctionFeaturizer(exclamation)),
                #("capitalization", FunctionFeaturizer(capitalizationRatio)),
                ("vectorizer", TfidfVectorizer( token_pattern=r'\b\w+\b', ngram_range=(1,2)))
                  ])
    # fit the above transformers to the data

    X_headline = union.fit_transform(X_headline)


    #reviewBody = pd.DataFrame(dataset, columns = ["review_body"])
    
    X_body = X["review_body"]
    bodyUnion = FeatureUnion([
        ("emojis", FunctionFeaturizer(emojis)),
        #("count_exclamation_mark", FunctionFeaturizer(exclamation)),
        #("length", FunctionFeaturizer(length)),
        ("capitalization", FunctionFeaturizer(capitalizationRatio)),
        ("vectorizer", TfidfVectorizer( token_pattern=r'\b\w+\b'))#, ngram_range=(1,2)))
    ])
    X_body = bodyUnion.fit_transform(X_body)


    X_votes = X["helpful_votes"]
    b = X_votes.as_matrix()
    X_votes = pd.DataFrame(b)

   # print(X_body)

    X_result = hstack([X_headline, X_body]).toarray()


    #####################################
    ## Training with cross validation
    #####################################
    scores = cross_validate(classifier, X_result, y.ravel(), scoring=scoringMetrics, cv=5, return_train_score=False)

    finish_time = time.time()
    elapsed_time = round(finish_time-start_time, 1)

    #Performance Metrics: f1, Precision, Recall
    avgf1 = round(statistics.mean(scores['test_f1_macro']), 3)
    avgprec = round(statistics.mean(scores['test_precision_macro']), 3)
    avgrecall = round(statistics.mean(scores['test_recall_macro']), 3)

    #print(datetime.datetime.now())
    print( "Classifier: " + classifier_name + "\tTesting Feature: " + testingFeatures + "\tIs Binary: " + str(binary_classification))
    print( "Precision Score: " + str(avgprec))
    print( "Recall Score: " + str(avgrecall))
    print( "f1 Score: " + str(avgf1))
    print( "Elapsed Time: " + str(elapsed_time) + " seconds")


    if isPlottingConfusionMatrix:
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

def capitalizationRatio(text):
    return len(re.findall("[A-Z]", text))/ len(text)

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