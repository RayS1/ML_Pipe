# Import libraries
import json
import plotly
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# Function to tokenize an input string
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response_clean', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    # Initialize lists
    category_names = []
    category_percents = []

    # Append lists for each new category
    for column_name in df.columns[4:]:
        category_names.append(column_name)
        count_false = df[column_name].value_counts()[0]
        count_true = df[column_name].value_counts()[1]
        category_percents.append(
            round((count_false - count_true) /
                  (count_false + count_true) * 100)
        )

    # Create dataframe
    categories_df = pd.DataFrame({'catagory_name': category_names,
                                  'category_percent': category_percents})
    
    categories_df = categories_df.sort_values(by='category_percent', ascending=False)
                      
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_percents
                )
            ],

            'layout': {
                'title': 'Category-wise Imbalance % <br>(Unflagged - Flagged) / Total',
                'yaxis': {
                    'title': "Imbalance Percentage(%)"
                },
                'xaxis': {
                    'title': "Category Name"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()