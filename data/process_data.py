# Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This method reads messages and categories csv files per supplied datafile names and
    returns a concatenated dataframe.
    Inputs:
    - messages_filepath - Messages csv file path
    - categories_filepath - Categories csv file path
    Returns:
    - Concatenated dataframe
    """
    # Read messages csv file into messages dataframe
    messages = pd.read_csv(messages_filepath)

    # Read categories csv file into categories dataframe
    categories = pd.read_csv(categories_filepath)

    # Concatenate messages and categories dataframes
    df = messages.merge(categories, how='inner', left_on='id', right_on='id')
    
    # Return the concatenated dataframe df
    return df
    
    

def clean_data(df):
    """
    This method cleans input dataframe df by splitting categories columns, by splitting 
    and converting to 36 category columns, each having binary values - 0 or 1.
    Inputs:
    - df - dataframe
    Returns:
    - Cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    category_df = df.categories.str.split(';', n=36, expand=True)

    # select the first row of the categories dataframe
    row = category_df.iloc[0]

    # use this row to extract a list of new column names for categories.
    # Apply a lambda function that takes everything up to the second to last character 
    # of each string with slicing
    category_colnames = row.apply(lambda c: c.split('-')[0])
    
    # rename the columns of `categories`
    category_df.columns = category_colnames
    
    for column in category_df:
        # set each value to be the last character of the string
        category_df[column] = category_df[column].apply(lambda c: c.split('-')[1])
    
        # convert column from string to numeric
        category_df[column] = category_df[column].astype('int')


    # For related category column there are three value 0, 1, 2. Replace each value 2 by 1.
    category_df['related'].replace(2, 1, inplace=True)

    # Category column child_alone contains only only value 0.
    # This column lacks predictability. Thus we drop child_alone category column.
    category_df.drop('child_alone', axis=1, inplace=True)
    
    # Drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(category_df, how='inner')

    
    # Drop duplicate rows from df dataframe 
    df = df.drop_duplicates()
    
    # Return the cleaned dataframe df
    return df



def save_data(df, database_filepath):
    """
    This method writes contents of the dataframe df into a sqlite database with user 
    supplied database filepath.
    Inputs:
    - df - input dataframe name
    - database_filepath - name of the sqlite datase.
    Returns:
    - None
    """
    # Open sqlite database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Save the cleansed dataframe df to the sqlite database into a table
    df.to_sql('disaster_response_clean', engine, index=False, if_exists='replace')
    
    # Return none
    return None
 


def main():
    if len(sys.argv) == 4:
        # Accept input parameters from the command line
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
              
        # Invoke load_data to read data from filenames provided
        df = load_data(messages_filepath, categories_filepath)

        # Clean data into df data frame
        print('Cleaning data...')
        df = clean_data(df)
        
        # Save cleansed data into database
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        # Throw informative message when user provides incomplete information
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()