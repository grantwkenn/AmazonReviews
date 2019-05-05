import csv
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot
import scikitplot
import random
import time
import numpy
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from imblearn.under_sampling import RandomUnderSampler


import configuration

#TODO IN PROGRESS  helpful_votes, total_votes, vine, verified_purchase
#TODO guessed star vs actual difference
#TODO more test files
#TODO IN PROGRESS cross validation -> grant
#TODO more features, nouns, adjectives
#TODO error analysis, highest magnitude
#TODO output distrubution for tsting

#########################################################
# UNIGRAM
#########################################################
def createUnigramVector(dataset, selectedFeature, X_res, y_res):   
    #choose vectorizer
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
    #vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

    #vectorize document
    data_vector = vectorizer.fit_transform(X_res[selectedFeature])
    return data_vector

##########################################################
#Plot confusion Matrix
##########################################################
def plotConfusionMatrix(classifier, data_vector, data_y):
    predictions = cross_val_predict(classifier, data_vector, data_y.ravel(), cv = 10)
    scikitplot.metrics.plot_confusion_matrix( data_y.ravel(), predictions, normalize=True)
    matplotlib.pyplot.savefig("save.png")
    matplotlib.pyplot.show()


##########################################################
#Generate Dataset, Undersample, Test
##########################################################

def testDataset(classifier, testingFeatures, dataType, selectedDataset, isPlottingConfusionMatrix):
    classifier_name = classifier[0]
    classifier = classifier[1]

    if ( dataType == "boolean" ):
        binary_classification = True
    else:
        binary_classification = False

    #input data from CSV file
    input_data = pd.read_csv(selectedDataset)

    input_data['review_headline_body'] = input_data.review_headline + ' ' + input_data.review_body

    #define dataset
    dataset = {"review_body": input_data["review_body"].values,
        "star_rating": input_data["star_rating"].values,
        "review_headline": input_data["review_headline"].values,
        "vine": input_data["vine"].values,
        "helpful_votes": input_data["helpful_votes"].values,
        "total_votes": input_data["total_votes"].values,
        "review_headline_body" : input_data["review_headline_body"].values }
    dataset = pd.DataFrame(data = dataset)
    dataset = dataset.dropna() #drop missing values


    if binary_classification: # omit 3 star reviews, binary classification
        dataset["label"] = dataset["star_rating"].apply(lambda rating : +1 if str(rating) > '3' else -1)
    else:
        dataset["label"] = dataset["star_rating"] # star classification

    tableFields = ["review_headline", "review_body", "star_rating", "helpful_votes", "total_votes", "review_headline_body"]
    X = pd.DataFrame(dataset, columns = tableFields)
    y = pd.DataFrame(dataset, columns = ["label"])

    #random undersample
    rus = RandomUnderSampler(random_state=13)
    X_res, y_res = rus.fit_resample(X, y)
    X_res = pd.DataFrame(X_res, columns = tableFields)

    dataVector = createUnigramVector(dataset, testingFeatures, X_res, y_res)
    
    #TODO REMOVE THIS LATER, WAS USING TO TEST APPENDING VECTORS
    testing_case = "review_headline"
    dataVector2 = createUnigramVector(dataset, testing_case, X_res, y_res )
    #TODO REMOV THIS AFTER WORKING 

    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    scores = cross_validate(classifier, dataVector, y_res.ravel(), cv = 10, scoring = scoring)
    avgf1 = np.mean(scores['test_f1_macro'])
    avgprecision = np.mean(scores['test_precision_macro'])
    avgrecall = np.mean(scores['test_recall_macro'])

    print("***********************************")
    print('Classifier: ' + testingFeatures)
    print("Scores: ", end = '')
    print(scores)
    print("***********************************")
    print("Average F1: ", end = '')
    print(avgf1)
    print("Average Precision: ", end = '')
    print(avgprecision)
    print("Average Recall: ", end = '')
    print(avgrecall)

    if isPlottingConfusionMatrix:
        plotConfusionMatrix(classifier, dataVector, y_res)