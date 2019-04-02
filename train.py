import csv
import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler

#input data from CSV file
###input_data = pd.read_csv("Amazon Review Datasets/video_games_truncated.csv")
input_data = pd.read_csv("Amazon Review Datasets/amazon_reviews_Video_Games.csv")
#input_data['star_rating'] = input_data['star_rating'].astype(object)
#input_data['review_body'] = input_data['review_body'].astype(object)

#dataset has two fields: review_body and star_rating
dataset = {"review_body": input_data["review_body"], "star_rating": input_data["star_rating"] }
dataset = pd.DataFrame(data = dataset)
dataset = dataset.dropna() #drop missing values

# omit 3 star reviews, binary classification
#dataset = dataset[dataset["star_rating"] != 3] # need datatype=object
dataset["label"] = dataset["star_rating"].apply(lambda rating : +1 if str(rating) > '3' else -1)

X = pd.DataFrame(dataset, columns = ["review_body"])
y = pd.DataFrame(dataset, columns = ["label"])

rus = RandomUnderSampler(random_state=13)
X_res, y_res = rus.fit_resample(X, y)

X1 = pd.DataFrame(X_res, columns = ["review_body"])


train_X, test_X, train_y, test_y = train_test_split(X1, y_res, random_state=50)
train_y=train_y.astype('int')

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_vector = vectorizer.fit_transform(train_X['review_body'])
test_vector = vectorizer.transform(test_X['review_body'])

######   FOR LOGISTIC REGRESSION
#clr = LogisticRegression()
#clr.fit(train_vector, train_y.ravel())
#scores = clr.score(test_vector, test_y) # accuracy


######  FOR SVM
clf = SVC(gamma='auto')
clf.fit(train_vector, train_y.ravel())
scores = clf.score(test_vector, test_y) # accuracy

print(scores)