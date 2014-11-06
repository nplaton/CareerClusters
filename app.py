from flask import Flask
from flask import request
import pandas as pd
import numpy as np
import pickle
import ipdb
from scipy import spatial

app = Flask(__name__)
 
fit_model = pickle.load(open( 'data/fitted_model.pkl', 'rb' ))
fit_vectorizer = pickle.load(open( 'data/fitted_vectorizer.pkl', 'rb' ))
vector_matrix = pickle.load(open( 'data/vector_matrix.pkl', 'rb' ))
df = pd.read_csv('data/SOpostings.csv')

# home page
@app.route('/')
def index():
    return '''
    <h1> Would you like a job? </h1>
    <h2> Enter some text, either your resume, description of a job you'd like, or any keywords </h2>
    <form action="/classifier" method='POST' >
        <input type="text" name="user_input" />
        <input type="submit" />
    </form> '''

# My classifier app
#==============================================

# create the page the form goes to
@app.route('/classifier', methods=['POST'] )
def classifier():
    # get data from request form, the key is the name you set in your form
    data = str(request.form['user_input'])
    vectorize_data = fit_vectorizer.transform([data])
    # guess = fit_model.predict(vectorize_data)
    # ipdb.set_trace()
    cos = []
    transformed = vectorize_data.T.toarray()
    vec_array = vector_matrix.toarray()
    for x in range(vec_array.shape[0]):
        cos.append(spatial.distance.cosine(vec_array[x,:], transformed))
    # for x in np.array(cos).argsort()[:10]:
    #     df.links[x]
    x = np.array(cos).argsort()[0]


    returner = ''' Top job for you: <br> <a href="%s" target="_blank"> %s </a>  <br> <a href="/"> Back to home page </a>''' 
    
    # now return your results 
    return  returner % (df.links[x], df.titles[x])

# fit_model.close()
# fit_vectorizer.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)