import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('data/SOpostings.csv')

desc = df.descriptions

vectorizer = TfidfVectorizer(stop_words = 'english', min_df=1)
vectorized_matrix = vectorizer.fit_transform(desc)

km = KMeans(7)
km.fit(vectorized_matrix)

pickle.dump(km, open('data/fitted_model.pkl', 'wb'))
pickle.dump(vectorizer, open('data/fitted_vectorizer.pkl', 'wb'))