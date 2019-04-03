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
binary_classification = False

use_all_classifiers = True
classifier_choice = "svc"


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
    print( classifier_choice + ": " + str(scores))


#########################################################
# RUN 
#########################################################

if use_all_classifiers:
    for classifier in classifiers:
        classifier_choice = classifier[0]
        clf = classifier[1]
        testDataset()
else:
    clf = findClassifier(classifier_choice)
    testDataset()