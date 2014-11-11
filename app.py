from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import ipdb
from scipy import spatial
import re
import resume_optimize
import json

app = Flask(__name__)
 
fit_model = pickle.load(open( 'data/fitted_model.pkl', 'rb' ))
fit_vectorizer = pickle.load(open( 'data/fitted_vectorizer.pkl', 'rb' ))
vector_matrix = pickle.load(open( 'data/vector_matrix.pkl', 'rb' ))
df = pd.read_csv('data/SOpostings.csv')

# home page
@app.route('/')
def index():
    return render_template('homepage.html')

# My classifier app
#==============================================

# create the page the form goes to
@app.route('/classifier', methods=['POST'] )
def classifier():
    # get data from request form, encode unicode and lose punctuation
    data = re.sub('[^\w\s]+', '', request.form['user_input']).encode('utf-8')
    vectorize_data = fit_vectorizer.transform([data])

    cos = []
    transformed = vectorize_data.T.toarray()
    vec_array = vector_matrix.toarray()
    for x in range(vec_array.shape[0]):
        cos.append(spatial.distance.cosine(vec_array[x,:], transformed))

    x = np.array(cos).argsort()[:20]
    returner_df = df.ix[x,:]
    data = returner_df.values
    
    # now return your results 
    return render_template('job_postings.html', data=data)

@app.route('/optimize_resume')
def optimize_resume():
    return render_template('resume_optimize.html')

@app.route('/improve_resume', methods=['POST'] )
def improve_resume():
    job_posting = re.sub('[^\w\s]+', '', request.form['jd']).encode('utf-8')
    print job_posting
    cv = re.sub('[^\w\s]+', '', request.form['cv']).encode('utf-8')
    data = resume_optimize.make_posting_to_dict(job_posting, cv)
    return render_template('resume_chart.html', data=data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)