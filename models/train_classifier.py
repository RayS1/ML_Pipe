# import required libraries
import sys
import sqlalchemy as db
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import classification_report #, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import re
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# constants defined here
RANDOM_STATE = 42
VERBOSE = 0 # Set it to 8 for most details
TEST_SIZE = 0.3

# Define stopwords and lemmatizer
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    """
    Method to load data from sqlite database.
    Inputs:
    - database_filepath: file specification of the database file
    Returns:
    - X: X containing messages column
    - y: 35 binary categories (one less than 36 because column child_only is eliminated)
    - category_names: Names of category columns
    """
    # Get handle to sql engine of the database contining cleaned disaster response data
    engine = db.create_engine('sqlite:///' + database_filepath)
    connection = engine.connect()
    
    # Read into a dataframe
    df = pd.read_sql_table('disaster_response_clean', connection)
    
    # Retrieve the category column names
    category_names = df.iloc[0][4:].index.tolist()

    # Create X and y values
    X = df[df.columns[1:4]]
    y = df[df.columns[4:]]
    
    # Return X and y
    return X, y, category_names


def tokenize(text):
    """
    Method to transform a message text into tokens.
    Inputs:
    - text - text message
    Returns:
    - tokens
    """
    
    # Normalize text and remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Return tokens
    return tokens
    

def build_model():
    """
    Method to determine optimal model from pipeline and parameters
    Inputs: 
    - None
    Returns:
    - Optimal model returned by Grid Search
    """

    # Create a pipeline using LinearSVC as classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC()))
    ])
    
    # Define hyper-parameter grid to be searched
    parameters = {
        'vect__binary': [True, False],
        'tfidf__use_idf': [True, False],
        'clf__estimator__class_weight': ['balanced', None]
    }

    # Perform grid search
    cv = GridSearchCV(pipeline,
                      param_grid=parameters,
                      scoring='f1_weighted',
                      cv=3,
                      n_jobs=4,
                      verbose=VERBOSE)

    # return the optimal model
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Method to evaluate a machine learning model
    Inputs:
    - model - machine learning model 
    - X_test - X matrix for testing
    - y_test - y matrix for testing
    - category_names - multi-classification category names
    Returns:
    - None
    Prints:
    - Best parameter information
    - Classification Report for various categories
    - Accuracy score
    """
    
    print("\nInfo: Best parameters:", model.best_params_, "\n\n")
    
    # Test the model
    y_pred = model.predict(X_test)
    # Convert predicted results to a dataframe
    y_pred = pd.DataFrame(y_pred)
    
    # Print classifier report, accuracy and confusion matrix
    print('\n\nClassification Report for various categories:\n')

    for i in range(len(category_names)):
        print('Category:', category_names[i])
        #print('Confusion matrix:\n', confusion_matrix(y_test[category_names[i]], y_pred[i]))
        print('Accuracy score:', accuracy_score(y_test[category_names[i]], y_pred[i]))
        print(classification_report(y_test[category_names[i]], y_pred[i]))


def save_model(model, model_filepath):
    """
    Method saves the model as a pickle file
    Inputs:
    - model - machine learning model
    - model_filepath - filepath where the machine learning model will be stored as pickle file
    Returns:
    - None
    """
    
    # Save model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():

    if len(sys.argv) == 3:
    
        # Retrieve database filepath from command line arguments
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        # Load data into X, y, and category_names variables
        X, y, category_names = load_data(database_filepath)
        
        # Only the message column is used for building the model
        X = X['message']
        
        # Create train_test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
        
        # Build the model
        print('Building model...')
        model = build_model()
        
        # Training the model
        print('Training model...')
        model.fit(X_train, y_train)
        
        # Evaluate the model
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        # Save the model
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        # Print informative message when user does not provide enough information
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
