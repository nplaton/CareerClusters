import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('data/SOpostings.csv')

desc = df.descriptions

vectorizer = TfidfVectorizer(stop_words = 'english', min_df=1)
vector_matrix = vectorizer.fit_transform(desc)

km = KMeans(7)
km.fit(vector_matrix)

pickle.dump(km, open('data/fitted_model.pkl', 'wb'))
pickle.dump(vectorizer, open('data/fitted_vectorizer.pkl', 'wb'))
pickle.dump(vector_matrix, open('data/vector_matrix.pkl', 'wb'))