import requests
from bs4 import BeautifulSoup
from HTMLParser import HTMLParser
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.corpus import stopwords
import re
import ipdb

# Get all links from StackOverflow
class ClusterWords(object):

    def __init__(self, query, n_words, n_clusters=7):
        self.query = query
        self.n_words = n_words
        self.n_clusters = n_clusters        

    def get_links_from_SO(self):
        links = []
        query = self.change(self.query)
        for i in range(1,40):
            base = 'http://careers.stackoverflow.com/jobs/tag/{0}?pg={1}'.format(query,i)
            r = requests.get(base)
            soup = BeautifulSoup(r.text, "html.parser")
            name = soup.find_all('h3', class_ = '-title')
            verybase = 'http://careers.stackoverflow.com'
            for a in name:
                links.append(verybase + str(a).split('href')[1].split()[0][1:].strip('"'))
        return links

    # Turn links into df
    def get_SO_postings_make_df(self):
        links = self.get_links_from_SO()
        titles_for_df = []
        links_for_df = []
        descriptions_for_df = []
        for link in links:
            r = requests.get(link)
            soup = BeautifulSoup(r.text, "html.parser")
            key = str(soup.title)[7:-34].replace('.', '')
            first_clean_content = ''.join([p.get_text(strip=True) for p in soup.find_all("div", "description")])
            clean_again = str(re.sub('[^\w\s]+', '', first_clean_content))[15:]
            description_dict = {'description': clean_again}
            titles_for_df.append(key)
            links_for_df.append(link)
            descriptions_for_df.append(clean_again)
            df = pd.DataFrame({'titles': titles_for_df, 'links': links_for_df, 'descriptions': descriptions_for_df})
            df.to_csv('data/SOpostings.csv', index=False)
        return df

    # Vectorize words
    def vectorize(self):
        data = self.get_SO_postings_make_df()
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1,3))
        descriptions = list(data['descriptions'].values)
        vector_matrix = vectorizer.fit_transform(descriptions)
        pickle.dump(vectorizer, open('data/fitted_vectorizer.pkl', 'wb'))
        pickle.dump(vector_matrix, open('data/vector_matrix.pkl', 'wb'))
        return vectorizer, vector_matrix

    # Init kmeans
    def init_kmeans(self): 
        vec, vector_matrix = self.vectorize()
        km = KMeans(self.n_clusters)
        km.fit(vector_matrix)
        pickle.dump(km, open('data/fitted_model.pkl', 'wb'))
        return km

    # Find words that most describe clusters
    def get_descriptive_words(self): #get most descriptive words for cluster
        descriptive_words = defaultdict(list)
        v, vm = self.vectorize()
        km = self.init_kmeans()
        for i in range(self.n_clusters):
            indices = np.argsort(km.cluster_centers_[i])[::-1][:self.n_words]
            for q in indices:
                descriptive_words[i].append(v.get_feature_names()[q])
        return descriptive_words.values()

    # Helper to format strings for scraping
    def change(self, string):
        if len(string.split()) > 1:
            return string.replace(' ', '-')
        return string

def main():
    query = 'Data Scientist'
    cw = ClusterWords(query, 2)
    print cw.get_descriptive_words()

if __name__ == '__main__':
    main()

