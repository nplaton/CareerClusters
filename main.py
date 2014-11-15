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

'''This file is meant to be run each night to scrape job postings from multiple sites and convert 
them into a format CareerClusters can use. The format of the file will remain the same for any 
number of job sources, but each site has unique html elements that need to be encoded in the
functions below'''

class ClusterWords(object):

    def __init__(self, query, n_words, n_clusters=7):
        self.query = query
        self.n_words = n_words
        self.n_clusters = n_clusters        

    # Get links from Stack Overflow Careers
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

    # Turn SO links into df
    def get_SO_postings_make_df(self):
        links = self.get_links_from_SO()
        titles_for_df = []
        descriptions_for_df = []
        for link in links:
            r = requests.get(link)
            soup = BeautifulSoup(r.text, "html.parser")
            key = str(soup.title)[7:-34].replace('.', '')
            first_clean_content = ''.join([p.get_text(strip=True) for p in soup.find_all("div", "description")])
            clean_again = str(re.sub('[^\w\s]+', '', first_clean_content))[15:]
            titles_for_df.append(key)
            descriptions_for_df.append(clean_again)
        df = pd.DataFrame({'titles': titles_for_df, 'links': links, 'descriptions': descriptions_for_df})
        return df

    #Get links from indeed.com
    def get_links_from_indeed(self):
        links = []
        query = 'data+scientist'
        verybase = 'http://www.indeed.com/viewjob?jk='
        for i in range(20):
            base = 'http://www.indeed.com/jobs?q={0}&l=San+Francisco%2C+CA&start={1}'.format(query,i*10)
            r = requests.get(base)
            soup = BeautifulSoup(r.text, "html.parser")
            name = soup.find_all('h2', class_ = 'jobtitle')
            for x in name:
                parts = str(x).split('jk=')
                if len(parts) > 1:
                    links.append(''.join(verybase + parts[1].split('"')[0]))
        return links

    # Turn Indeed links into df
    def get_indeed_postings_make_df(self):
        titles_for_df = []
        links_for_df = self.get_links_from_indeed()
        descriptions_for_df = []
        for link in links_for_df:
            r = requests.get(link)
            soup = BeautifulSoup(r.text, "html.parser")
            titles_for_df.append(soup.title.string[:-13])
            for td in soup.find_all('td', attrs={'class':'snip'}):
                cleaned = str(re.sub('[^\w\s]+', '', td.text)).replace('\n', '')
                descriptions_for_df.append(cleaned)
        df_indeed = pd.DataFrame({'titles': titles_for_df, 'links': links_for_df, 'descriptions': descriptions_for_df})
        df_indeed.descriptions = [x.encode('utf-8') for x in df_indeed.descriptions]
        df_indeed.titles = [x.encode('utf-8') for x in df_indeed.titles]
        df_indeed.links = [x.encode('utf-8') for x in df_indeed.descriptions]
        return df_indeed

    # Make df of postings from all sources and write to file
    def make_final_df(self):
        df_SO = self.get_SO_postings_make_df()
        df_indeed = self.get_indeed_postings_make_df()
        big_df = pd.concat((df_SO, df_indeed))
        big_df['dropped'] = [len(x) > 100 for x in big_df['descriptions']]
        big_df = big_df[big_df.dropped == 1] #If posting too short, drop row
        big_df = big_df[[x for x in big_df.columns if x != 'dropped']]
        big_df.to_csv('data/SOpostings.csv', index=False)
        return big_df 

    # Vectorize words
    def vectorize(self):
        data = self.make_final_df()
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

