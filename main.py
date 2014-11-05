import requests
from bs4 import BeautifulSoup
from HTMLParser import HTMLParser
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.corpus import stopwords

# Get all links from StackOverflow
num = 0
all_info = {}
links = []

def get_links_from_SO(query):
	query = change(query)
	for i in range(1,42):
	   base = 'http://careers.stackoverflow.com/jobs/tag/{0}?pg={1}'.format(query,i)
	   r = requests.get(base)
	   soup = BeautifulSoup(r.text, "html.parser") 
	   # print soup
	   name = soup.find_all('h3', class_ = '-title')
	   verybase = 'http://careers.stackoverflow.com'
	   for a in name:
	       links.append(verybase + str(a).split('href')[1].split()[0][1:].strip('"'))

# Turn links into postings
def get_SO_postings_make_dict(links):
	for l in links:
	   r = requests.get(l)
	   soup = BeautifulSoup(rr.text, "html.parser")
	   all_info[d] = ''.join([p.get_text(strip=True) for p in soup.find_all("div", "description")])

# Vectorize words
def vectorize(data): 
   vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1,2))
   vector_matrix = vectorizer.fit_transform(data)
   return vectorizer, vector_matrix

# Init kmeans
def init_kmeans(vector_matrix, n): 
   km = KMeans(n)
   km.fit(vector_matrix)
   return km

# Find words that most describe clusters
def get_descriptive_words(feature_matrix, n_clusters, n_words): #get most descriptive words for cluster
   descriptive_words = defaultdict(list)
   v, vm = vectorize(feature_matrix)
   km = init_kmeans(vm, n_clusters)
   for i in range(len(km.cluster_centers_)):
       indices = np.argsort(km.cluster_centers_[i])[::-1][:n_words]
       for q in indices:
           descriptive_words[i].append(v.get_feature_names()[q])
   return descriptive_words.values()

all_info.update((x, str(re.sub('[^\w\s]+', '', y))) for x, y in all_info.items())

# Helper to format strings for scraping
def change(string):
    if len(string.split()) > 1:
        return string.replace(' ', '-')
    return string

def main(x, y, z):
	return get_descriptive_words(x, y, z)

if __name__ == '__main__':
	main()

