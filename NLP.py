import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from streamlit.logger import get_logger
from sklearn.decomposition import TruncatedSVD
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from nltk.corpus import stopwords

enc = LabelEncoder()
# Load data.
data = pd.read_csv("data_final.csv", index_col=0, usecols=[0,1,2,4])
data = data.dropna()
samples = data.transcription
text_labels  = [label_name.lower() for label_name in data.medical_specialty]
labels = enc.fit_transform(np.array(text_labels))
# Transform data.
max_df = 0.5
min_df = 0.001
max_features = 1000
ngram_range = (1,1)
final_stopwords_list = stopwords.words('french')
tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, stop_words=final_stopwords_list, ngram_range=ngram_range)
tfidf = tfidf_vectorizer.fit_transform(samples)
feature_names = tfidf_vectorizer.get_feature_names()
# Title.
st.sidebar.header("Constructing dictonary of words.")
# Upper bound for tf-idf value.
max_df = 0.3
# Lower bound for tf-idf value.
min_df = 0.01
# Size of dictionary.
max_features = 500
# Dimensionality reduction.
dim_red = TruncatedSVD(n_components=2)
data_red = dim_red.fit_transform(tfidf)

# Number of trees.
n_estimators = 1000
# Define classifier.
forest_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, max_leaf_nodes=None, class_weight='balanced', oob_score=True, n_jobs=-1, random_state=0)
# Define grid.
parameters = {'max_leaf_nodes':np.linspace(20,35,14,dtype='int')}
# Balanced accuracy as performance measure.
clf = RandomizedSearchCV(forest_clf, parameters, n_iter=10, cv=3, scoring='accuracy',n_jobs=-1)
# Train/optimize classifier.
classifier = clf.fit(tfidf, labels)
# Retrieve optimum.
forest = classifier.best_estimator_
feature_importances = forest.feature_importances_
indices = np.argsort(feature_importances)[::-1]
st.sidebar.header("Customizing the model.")
n_estimators = 1000
max_leaf_nodes = 25
max_depth = 5
class_weight = 'balanced_subsample'
# Retrieve values.
y_true = labels
y_pred = classifier.predict(tfidf)
# Compute scores.
f1_score_ = f1_score(y_true,y_pred,average="weighted")
cm = confusion_matrix(y_true,y_pred)
print(f1_score_)
print(cm)