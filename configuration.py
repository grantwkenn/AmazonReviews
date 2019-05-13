from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, NuSVC

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
votingClassifier = VotingClassifier(estimators= [ 
    ("nu_svc" , NuSVC(gamma = 'auto')),
    ("decision_tree" , DecisionTreeClassifier(max_depth=5)),
    ("random_forest" , RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ("mlp" , MLPClassifier(alpha=1)),
    ("nb" , MultinomialNB())
    ], voting='hard')

classifiers = [
    ("nu_svc" , NuSVC(gamma = 'auto')),
    ("svc" , SVC(gamma = 'auto') ),
    ("svc2" , SVC(kernel="linear", C=0.025)),
    ("decision_tree" , DecisionTreeClassifier(max_depth=5)),
    ("random_forest" , RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ("mlp" , MLPClassifier(alpha=1)),
    ("nb" , MultinomialNB()),
    ("votingClassifier", votingClassifier)
]


#scoring metrics
scoringMetrics = ['precision_macro', 'recall_macro', 'f1_macro']

#possible csv
video_game_data = "Amazon Review Datasets/video_games_truncated.csv"
#video_game_data = "Amazon Review Datasets/vg_trunc_90k.csv"
kitchen_data = "Amazon Review Datasets/kitchen_truncated.csv"

#possible options
options = [
    "review_headline",
    #"review_body"
    #"combined"
]

#possible types of classification
catagory_types = [
    "boolean",
    "catagories"
]

#possible labels
star_label = "star_rating"

#features being selected
testing_options = ["review_headline"]
testing_labels = star_label
selected_dataset = video_game_data
selected_classifiers = ["nb"]
testing_types = ["catagories"]

plot_confusion_matrix = False

#use all options
use_all_options = False
use_all_classifiers = False
use_all_types = False


############################################################
# END CONFIGURATION SET UP
############################################################
testing_classifiers = []

for classifier in selected_classifiers:
    testing_classifiers.append(findClassifier(classifier))

if use_all_options:
    testing_options = options

if use_all_classifiers:
    testing_classifiers = classifiers

if use_all_types:
    testing_types = catagory_types

def getTestingClassifiers():
    return testing_classifiers

def getTestingOptions():
    return options

def getTestingTypes():
    return testing_types

def getSelectedDataset():
    return selected_dataset

def getPlotSetting():
    return plot_confusion_matrix

def getScoringMetrics():
    return scoringMetrics