# import libraries
import pandas as pd
from sqlalchemy.engine import create_engine

import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

import sys
import os

def load_data():
    """
    This function loads the X & Y data from the cleaned database.
    
    Parameter:
    The name of this database is from the command line input.
    
    Returns:
    X (dataframe): Disaster relief messages.
    Y (dataframe): The multiple classifications of the above disaster relief messages.

    """


    # load data from database
    #path = r'D:\OneDrive\03-Learning\01-Online\Udacity-Data-Science-ND\100-Projects\02-Disaster-Response-Pipeline\data\disaster_response_cleaned.db'
    #path = path.replace('\\','\\\\')
    
    cwd = os.getcwd()
    input_db = sys.argv[1]
    path = cwd + '\\' + str(input_db)
    path = path.replace('\\','\\\\')
    print(path)

    sql_path = f"sqlite:///{path}"
    engine = create_engine(sql_path)
    df = pd.read_sql('SELECT * FROM disaster_response_cleaned',con=engine)

    X = df['message']
    Y = df.iloc[:,4:]
    
    return X,Y

def tokenize(text):

    """
    Carries out a series of transforms on the text provided in order to break it down to a format suitable for vectorization.
    
    Parameter:
    text (str): A disaster relief message to be tokenized.
    
    Returns:
    clean_tokens (list): A list of tokens representing a version of the text that is suitable for vectorization.
    """
    
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    tokens = word_tokenize(text)
    lemmatizer =  WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token,pos='v')
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    """
    Builds an NLP pipeline, defines the parameters of the grid search and preps the model 
    (training TBD, need to run .fit() on output of this function).
    
    Returns:
    cv (sklearn.model_selection._search.GridSearchCV): A GridSearchCV object defining the NLP pipeline and parameters of the grid search.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    
    params = {'clf__estimator__n_estimators': [10, 25],
              'clf__estimator__min_samples_split': [2, 4]}
    
    cv = GridSearchCV(pipeline,param_grid=params)
    
    return cv

def score_model(model,X_test,y_test):
    """
    Scores the NLP model.
    
    Parameter:
    model: The trained model that is to be assessed.
    X_test: The test subset of the input variables (text in this case).
    y_test: The test subset of the output variables (multiple classifications in this case).
    
    Returns:
    A print out of sklearn.metrics classification report (precision, recall, f1-score, support) for each category of the output.
    """
    
    y_pred = model.predict(X_test)
    
    print('\n\nCLASSIFICATION REPORT\n\n')

    for i,col in enumerate(y_test.columns):

        yt = y_test[col] 
        yp = y_pred[:,i]

        print('\n----------------\n')
        print(f"Classification Report for '{col}':\n")
        print(classification_report(yt,yp))
        print('\n----------------\n')
        
def save_model(model,model_filepath):
    """
    Save the trained NLP model.
    
    Parameter:
    model: The trained model that is to be saved.
    model_filepath (str): Filepath for the model with the name for the model - should end with .pkl.
    
    Returns:
    Dumps the model into a .pkl file in the location specified.
    """
    with open(model_filepath,'wb') as f:
        pickle.dump(model, f)
    
    
def main():
    """
    Stacks up all the previous functions in the right order, runs them and creates a model.
    """
    print('loading database')
    X,Y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    
    print('building model')
    model = build_model()
    
    print('training model')
    model.fit(X_train,y_train)
    
    print('scoring model')
    score_model(model,X_test,y_test)
    
    print('saving model')
    model_filepath = 'classifier.pkl'
    save_model(model,model_filepath)
    print('model saved')
    

if __name__ == '__main__':
    main()