# Disaster Response Pipeline Project

### Table of Contents

1. [Project Description](#project)
2. [Instructions](#instructions)
3. [Requirements](#requirements)
4. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)
6. [References](#references)

## Project Description <a name="project"></a>

For this project, intended to help emergency workers with classification of disaster response messages. It has the following application components:
1. ETL Pipeline - that takes textual input and transforms it into clean data that could be fed to ML pipeline
2. ML Pipeline - takes the clean data and processes it through Machine Learning (ML) pipeline to determine multiple classification that applies to the disaster response message.
3. Web interface - takes user's textual input to display multiple classifications that applies to the message.

## Instructions <a name="instructions"></a>

There are a few Python libraries needed for running all the included notebooks. However, Anaconda distribution of Python includes those. The code shold run using Python version 3.*
Further details of Python package requirements are available here (#requirements).

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Requirements <a name="requirements"></a>

For this project few Python libraries have been used. Their version numbers are:
1. Python ............. 3.9.13
2. Numpy version ...... 1.12.1
3. Pandas version ..... 0.23.3
4. nltk ............... 3.2.5
5. Flask .............. 0.12.2
6. pytest ............. 4.5.0
7. Matplotlib ......... 2.1.0
8. SQLAlchemy ......... 1.1.13
9. scikit-learn ....... 0.19.1

## File Description <a name="files"></a>

The following is a list of files needed to run this application. These are listed under various categories:

### Data files
1. disaster_messages.csv - contains disaster relief messages 
2. disaster_categories.csv - contains categories for disaster_message file contents

### Python scripts
1. process_data.py - 
2. train_classifier.py - 
3. run.py - 

### html Files
1. go.html
2. master.html

### Other Files
These objects are created and maintained by application code as necessary.
1. Sqlite database
2. Cleaned data table
3. ML Model - pickle file

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
Must give credit to Figure8 for the Disaster response data. If you use this work, you must acknowledge it too. Otherwise, feel free to use the code here as you would like!

## References <a name="references"></a>
1. [Using GridSearchCV](https://towardsdatascience.com/using-gridsearchcv-76614defc594) by Andrew Cole
2. [A Gentle Introduction to the Bag-of-Words Model](https://machinelearningmastery.com/gentle-introduction-bag-words-model/) by Jason Brownlee
3. [How to Tune Algorithm Parameters with Scikit-Learn](https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/) by Yaokun Lin

