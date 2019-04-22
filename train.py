import csv
import pandas as pd
import string
import matplotlib.pyplot
import scikitplot
import random
import time
import numpy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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
#SYNTAX FOR FEATURE SELECTION
#To select a combination of features { "feature_one", "feautre_two"}
#To select multiple runs of features ["feature_one", "feature_two"]
#These can be combined
#######################################################################


testing_features = ["review_headline", "review_body"]
testing_labels = star_label
selected_dataset = video_game_data
selected_classifiers = ["nb"]
testing_types = ["boolean"]

plot_confusion_matrix = False

#use all options
use_all_features = False
use_all_classifiers = False
use_all_types = False

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
    
    testing_features = ("review_body", "review_headline")
    
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

    rus = RandomUnderSampler(random_state=random.seed(time.time()))
    X_res, y_res = rus.fit_resample(X, y)


    X1 = pd.DataFrame(X_res, columns = [testing_features])

    train_X, test_X, train_y, test_y = train_test_split(X1, y_res, random_state=random.seed(time.time()))
    train_y=train_y.astype('int')

    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    

    train_vector = vectorizer.fit_transform(train_X[testing_features])
    test_vector = vectorizer.transform(test_X[testing_features])

    classifier.fit(train_vector, train_y.ravel())
    scores = classifier.score(test_vector, test_y)

    result = "Classifier: " + classifier_name + "  Testing Feature: "
    for feature in testing_features:
        result += str(feature) 
    result += "\tIs Binary: " + str(binary_classification)
    result += "Score: " +  ": " + str(scores)
    print(result)

    if plot_confusion_matrix:
        plotConfusionMatrix(classifier, test_vector, test_y)


def plotConfusionMatrix(classifier, test_vector, test_y):
    predictions = cross_val_predict(classifier, test_vector, test_y.ravel(), cv = 3)
    scikitplot.metrics.plot_confusion_matrix( test_y.ravel(), predictions, normalize=True)
    matplotlib.pyplot.show()
  

#########################################################
# RUN 
#########################################################

for classifier in testing_classifiers:
    for feature in testing_features:
        for data_type in testing_types:
            testDataset(classifier, feature, data_type)