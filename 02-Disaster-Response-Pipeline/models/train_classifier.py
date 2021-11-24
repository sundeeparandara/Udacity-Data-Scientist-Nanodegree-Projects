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
    
    with open(model_filepath,'wb') as f:
        pickle.dump(model, f)
    
    
def main():
    
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