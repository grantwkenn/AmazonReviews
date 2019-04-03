import csv
import pandas as pd
import string
import matplotlib.pyplot
import scikitplot

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

#possible csv
video_game_data = "Amazon Review Datasets/video_games_truncated.csv"
kitchen_data = "Amazon Review Datasets/kitchen_truncated.csv"

#possible features
features = [
    "review_headline",
    "review_body"
    #"review_combined" #TODO combine these
]

#possible types of classification
catagory_types = [
    "boolean",
    "catagories"
]

#possible labels
star_label = "star_rating"

#features being selected
testing_features = ["review_body"]
testing_labels = star_label
selected_dataset = video_game_data
selected_classifiers = ["nb", "svc2"]
testing_types = ["boolean"]

plot_confusion_matrix = True

#use all options
use_all_features = True
use_all_classifiers = True
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

def testDataset(classifier, testing_feature, data_type):
    classifier_name = classifier[0]
    classifier = classifier[1]
    
    if ( data_type == "boolean" ):
        binary_classification = True
    else:
        binary_classification = False

    #input data from CSV file
    input_data = pd.read_csv(selected_dataset)

    #define dataset
    dataset = {"review_body": input_data["review_body"], "star_rating": input_data["star_rating"], "review_headline": input_data["review_headline"] }
    dataset = pd.DataFrame(data = dataset)
    dataset = dataset.dropna() #drop missing values

    if binary_classification: # omit 3 star reviews, binary classification
        dataset["label"] = dataset[star_label].apply(lambda rating : +1 if str(rating) > '3' else -1)
    else:
        dataset["label"] = dataset[star_label] # star classification

    X = pd.DataFrame(dataset, columns = [testing_feature])
    y = pd.DataFrame(dataset, columns = ["label"])

    rus = RandomUnderSampler(random_state=13)
    X_res, y_res = rus.fit_resample(X, y)


    X1 = pd.DataFrame(X_res, columns = [testing_feature])

    train_X, test_X, train_y, test_y = train_test_split(X1, y_res, random_state=50)
    train_y=train_y.astype('int')

    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')


    train_vector = vectorizer.fit_transform(train_X[testing_feature])
    test_vector = vectorizer.transform(test_X[testing_feature])

    classifier.fit(train_vector, train_y.ravel())
    scores = classifier.score(test_vector, test_y)

    print()
    print( "Classifier: " + classifier_name + "\tTesting Feature: " + testing_feature + "\tIs Binary: " + str(binary_classification))
    print( "Score: " +  ": " + str(scores))

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