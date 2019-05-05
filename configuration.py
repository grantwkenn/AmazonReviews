from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, NuSVC

###############################################################
# BEGIN CONFIGURATION SETUP
###############################################################

#possible classifiers
classifiers = [
    ("nu_svc" , NuSVC(gamma = 'auto')),
    ("svc2" , SVC(kernel="linear", C=.5)),
    ("decision_tree" , DecisionTreeClassifier(max_depth=5)),
    ("random_forest" , RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ("mlp" , MLPClassifier(alpha=1)),
    ("nb" , MultinomialNB())
]


def findClassifier (classifier_name):
    for element in classifiers:
        if element[0] == classifier_name:
            return element
    print("NO VALID CLASSIFIER CHOSEN!")
    quit()

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
    "vine",
    "verified_purchase"
]

#possible types of classification
catagory_types = [
    "boolean",
    "catagories"
]

##########################################################################
#Feature selection
#######################################################################

testing_features = ["review_body"]
testing_labels = "star_label"
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

def getTestingClassifiers():
    return testing_classifiers

def getTestingFeatures():
    return testing_features

def getTestingTypes():
    return testing_types

def getSelectedDataset():
    return selected_dataset

def getPlotSetting():
    return plot_confusion_matrix