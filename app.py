from flask import Flask
from flask import request
import pickle
import ipdb
app = Flask(__name__)
 
fit_model = pickle.load(open( 'data/fitted_model.pkl', 'rb' ))
fit_vectorizer = pickle.load(open( 'data/fitted_vectorizer.pkl', 'rb' ))

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
    guess = fit_model.predict(vectorize_data)
    # ipdb.set_trace()
    returner = ''' Category: goes here <br> <a href="/"> Back to home page </a>''' 
    
    # now return your results 
    return  returner

# fit_model.close()
# fit_vectorizer.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)