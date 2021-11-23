from flask import Flask
from flask import render_template, request

#import sklearn.external.joblib as extjoblib
#import joblib

import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

#load model
#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize
from train_classifier import tokenize


#https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
#if __name__ == '__main__':
print('loading model')
with open("../models/classifier.pkl", 'rb') as f:
	model = pickle.load(f)
	print('model loaded')
	#print(model.predict(["we need water"]))
	
#model = joblib.load("../models/classifier.pkl")

#to launch the home page
@app.route('/')
def home():
    return render_template('home.html')

#to launch the results page and show prediction
@app.route('/',methods=['POST'])
def send():
	if request.method == 'POST':
		myInput = request.form["myInput"]
		print(myInput)
		
		prediction = model.predict([myInput])
		
		#print(type(prediction))
		#print(prediction.shape)
		
		cols = ['related', 'request', 'offer', 'aid_related', 'medical_help',
		   'medical_products', 'search_and_rescue', 'security', 'military',
		   'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
		   'missing_people', 'refugees', 'death', 'other_aid',
		   'infrastructure_related', 'transport', 'buildings', 'electricity',
		   'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
		   'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
		   'other_weather', 'direct_report']
		
		#print(np.shape(cols))
		
		
		df_prediction = pd.DataFrame(data=prediction,columns=cols).transpose()

		print(df_prediction)
		
		print(prediction.reshape(36,))
		print(cols)
		
		dict_prediction = dict(zip(cols,prediction.reshape(36,)))
		
		print(dict_prediction)
		
		return render_template('results.html',myInput=myInput,dict_df_prediction=dict_prediction)


def main():
	app.run(host='localhost', port=8000, debug=True)
	
if __name__ == '__main__':
	main()