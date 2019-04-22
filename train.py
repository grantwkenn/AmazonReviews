import csv
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot
import scikitplot
import random
import time
import numpy
from sklearn.metrics import classification_report
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
#from sklearn.pipeline import FeatureUnion
#from sklearn.pipeline import PCA, truncatedSVD
#TODO IN PROGRESS  helpful_votes, total_votes, vine, verified_purchase
#TODO IN PROGRESS case, stem, stopwords --> GRANT
#TODO guessed star vs actual difference
#TODO more test files
#TODO IN PROGRESS cross validation -> grant
#TODO more features, nouns, adjectives
#TODO error analysis, highest magnitude
#TODO output distrubution for tsting
#TODO precision recall, f score -->
#TODO k-fold --> GRANT

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
    ("svc2" , SVC(kernel="linear", C=.5)),
    ("decision_tree" , DecisionTreeClassifier(max_depth=5)),
    ("random_forest" , RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ("mlp" , MLPClassifier(alpha=1)),
    ("nb" , MultinomialNB())
]

#possible csv
vg_trunc_20k = "Amazon Review Datasets/truncations/vg_trunc_20k.csv"
video_game_data = "Amazon Review Datasets/video_games_truncated.csv"
kitchen_data = "Amazon Review Datasets/kitchen_truncated.csv"

#possible features
features = [
    "review_headline",
    "review_body",
    "review_headline_body",
    "helpful_votes",
    "total_votes",
    "vine"
]

#possible types of classification
catagory_types = [
    "boolean",
    "catagories"
]

#possible labels
star_label = "star_rating"


##########################################################################
#Feature selection
#######################################################################

testing_features = ["review_body", "review_headline"]
testing_labels = star_label
selected_dataset = video_game_data
selected_classifiers = ["nb"]
testing_types = ["boolean"]

plot_confusion_matrix = False

#use all options
use_all_features = False
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
#Generate Dataset, Undersample, Test
##########################################################

def testDataset(classifier, testing_features, data_type):
    classifier_name = classifier[0]
    classifier = classifier[1]

    if ( data_type == "boolean" ):
        binary_classification = True
    else:
        binary_classification = False

    #input data from CSV file
    input_data = pd.read_csv(selected_dataset)

    input_data['review_headline_body'] = input_data.review_headline + ' ' + input_data.review_body

    #define dataset
    dataset = {"review_body": input_data["review_body"].values,
        "star_rating": input_data["star_rating"].values,
        "review_headline": input_data["review_headline"].values,
        "vine": str(input_data["vine"].values),
        "helpful_votes": str(input_data["helpful_votes"].values),
        "total_votes": str(input_data["total_votes"].values),
        "review_headline_body" : input_data["review_headline_body"].values }
    dataset = pd.DataFrame(data = dataset)
    dataset = dataset.dropna() #drop missing values


    if binary_classification: # omit 3 star reviews, binary classification
        dataset["label"] = dataset[star_label].apply(lambda rating : +1 if str(rating) > '3' else -1)
    else:
        dataset["label"] = dataset[star_label] # star classification

    
    X = pd.DataFrame(dataset, columns = [testing_features])
    y = pd.DataFrame(dataset, columns = ["label"])

    #random undersample
    rus = RandomUnderSampler(random_state=13)
    X_res, y_res = rus.fit_resample(X, y)
    X_res = pd.DataFrame(X_res, columns = [testing_features])

    #choose vectorizer
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
    #vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

    #vectorize document
    data_vector = vectorizer.fit_transform(X_res[testing_features])

    scoring = ['precision_macro', 'recall_macro', 'f1_macro']

    scores = cross_validate(classifier, data_vector, y_res.ravel(), cv = 10, scoring = scoring)
    avgf1 = np.mean(scores['test_f1_macro'])
    avgprecision = np.mean(scores['test_precision_macro'])
    avgrecall = np.mean(scores['test_recall_macro'])

    print("***********************************")
    print('Classifier: ' + testing_features)
    print("Scores: ", end = '')
    print(scores)
    print("***********************************")
    print("Average F1: ", end = '')
    print(avgf1)
    print("Average Precision: ", end = '')
    print(avgprecision)
    print("Average Recall: ", end = '')
    print(avgrecall)

    if plot_confusion_matrix:
        plotConfusionMatrix(classifier, data_vector, y_res)


def plotConfusionMatrix(classifier, data_vector, data_y):
    predictions = cross_val_predict(classifier, data_vector, data_y.ravel(), cv = 10)
    scikitplot.metrics.plot_confusion_matrix( data_y.ravel(), predictions, normalize=True)
    matplotlib.pyplot.savefig("save.png")
    matplotlib.pyplot.show()


#########################################################
# RUN
#########################################################

for classifier in testing_classifiers:
    for feature in testing_features:
        for data_type in testing_types:
            testDataset(classifier, feature, data_type)
