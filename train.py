import csv
import pandas as pd
import string

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
            return element[1]
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
    #("nb" , GaussianNB())
]

#possible csv
video_game_data = "Amazon Review Datasets/video_games_truncated.csv"
kitchen_data = "Amazon Review Datasets/kitchen_truncated.csv"

#possible features
features = [
    "review headline",
    "review_body",
    "review_combined" #TODO combine these
]

#possible labels
star_label = "star_rating"

#features being selected
testing_feature = "review_body"
testing_label = star_label
selected_dataset = video_game_data
classifier_choice = "svc"
binary_classification = False

#use all options
use_all_features = True
use_all_classifiers = True
test_binary_plus_catagory = True

clf = findClassifier(classifier_choice)

############################################################
# END CONFIGURATION SET UP
############################################################



##########################################################
#Generate Dataset, Undersample, Test
##########################################################

def testDataset():
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


    train_vector = vectorizer.fit_transform(train_X['review_body'])
    test_vector = vectorizer.transform(test_X['review_body'])

    clf.fit(train_vector, train_y.ravel())
    scores = clf.score(test_vector, test_y)
    print()
    print( "Classifier: " + classifier_choice + "\tTesting Feature: " + testing_feature + "\tIs Binary: " + str(binary_classification))
    print( "Score: " +  ": " + str(scores))
    

def test_all_classifiers():
    global classifier_choice
    global clf
    for classifier in classifiers:
        classifier_choice = classifier[0]
        clf = classifier[1]
        testDataset()

def test_all():
    global binary_classification
    binary_classification = False
    test_all_classifiers()
    binary_classification = True
    test_all_classifiers()

def test_binary_plus_catagory():
    global binary_classification
    binary_classification = False
    testDataset()
    binary_classification = True
    testDataset()


#########################################################
# RUN 
#########################################################

if test_binary_plus_catagory and use_all_classifiers:
    test_all()

elif use_all_classifiers:
    test_all_classifiers()

elif test_binary_plus_catagory:
    test_binary_plus_catagory()

else:
    testDataset()