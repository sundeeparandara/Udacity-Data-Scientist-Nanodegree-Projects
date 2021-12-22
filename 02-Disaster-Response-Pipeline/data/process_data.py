import sys
import pandas as pd
from sqlalchemy.engine import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads data from two files and merges them into a single dataframe.
    
    Parameters:
    messages_filepath (str): filepath to the csv containing disaster relief messages
    categories_filepath (str):  filepath to the csv containing the multi-classifications of the above messages
    
    Returns:
    df (dataframe): A dataset that contains the messages and the classifications.
    """
    #pass
    #messages = pd.read_csv('messages.csv')
    #categories = pd.read_csv('categories.csv')
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how='left',on='id')
    return df

def clean_data(df):
    #pass

    """
    Cleans the data into a format that is more suitable for processing further down the pipeline.
    
    Parameters:
    df (dataframe): A dataset that contains the messages and the classifications.
    
    Returns:
    df (dataframe): A cleaned version of the above dataset.
    """
    categories = df.categories.str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    df.drop(['categories'],axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df = df[df['related']!=2]
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):

    """
    Saves the data into a SQLite database
    
    Parameters:
    df (dataframe): A cleaned dataset.
    database_filename (str): The name of the database file to be created - should end in .db.  
    
    Returns:
    A database file with the above specified name.
    """

    #pass  
    #engine = create_engine('sqlite:///disaster_response_cleaned.db')
    #df.to_sql('disaster_response_cleaned', engine, index=False, if_exists='replace')
    engine_path = f'sqlite:///{database_filename}'
    print(f'engine path = {engine_path}')
    engine = create_engine(engine_path)
    table_name = database_filename.split('.')[0]
    print(f'table name : {table_name}')
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    """
    Grabs the required variables from the command line,
    stacks the above functions in the right order and generates a clean .db file.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()