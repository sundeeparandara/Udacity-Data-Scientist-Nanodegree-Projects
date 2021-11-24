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
import joblib


# load data from database
path = r'D:\OneDrive\03-Learning\01-Online\Udacity-Data-Science-ND\Udacity-Data-Scientist-Nanodegree-Projects\02-Disaster-Response-Pipeline\models\disaster_response_cleaned.db'
path = path.replace('\\','\\\\')
print(path)
sql_path = f"sqlite:///{path}"
print(sql_path)
engine = create_engine(sql_path)
df = pd.read_sql('SELECT * FROM disaster_response_cleaned',con=engine)
print('data loaded')

X = df['message']
Y = df.iloc[:,4:]

print('data extracted')

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


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
])

print('pipeline created')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print('training...')

pipeline.fit(X_train, y_train)

filename = 'pipeline_trained_model_joblib.sav'
joblib.dump(cv, filename)

print('predicting...')

y_pred = pipeline.predict(X_test)

print('\n\nCLASSIFICATION REPORT\n\n')

for i,col in enumerate(y_test.columns):
    
    yt = y_test[col] 
    yp = y_pred[:,i]

    print('\n----------------\n')
    print(f"Classification Report for '{col}':\n")
    print(classification_report(yt,yp))
    print('\n----------------\n')