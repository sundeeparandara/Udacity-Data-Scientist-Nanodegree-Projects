
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File & Folder Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run with no issues using Python versions 3.*.

The instructions below are how to run the app from Windows (has not been tested in other OS):

1. Download the entire repository.
2. Switch on Anaconda Command prompt (or equivalent) and browse to the 'app' folder.
3. Type "python app.py". 
4. After a few seconds, the following message should appear "Running on http://localhost:8000/".
5. Copy "http://localhost:8000/" to Chrome (or other browser).
6. Use the app to classify a disaster response messages.

## Project Motivation<a name="motivation"></a>

During times of crises, it is important for disaster response authorities to be able to respond effectively. The ability to respond effectively is based on the ability to accurately judge the category of the "call to aid".

This project builds a simple web-based app, with a NLP model in the back-end, to classify the messages.

The focus of the project is to get a basic app operational - the NLP model itself is a 2nd iteration based on a simple grid-search of just two parameters, and further work can be done to improve this. 


## File & Folder Descriptions <a name="files"></a>

There are three folders in the repository:

1. "app": This contains the app that was built for the project, and contains the "app.py" file.
2. "data": This contains the data provided by Figure Eight, the processing done to clean and integrate this data and is stored in a database file called "disaster_response_cleaned.db". "process_data.py" gives insight into the steps taken for the data cleaning integration.
3. "models": This contains the work done to derive the NLP model to help classify messages. "train_classifier.py" is the script that summarises the process to generate the classification model "classifier.pkl".

## Acknowledgements<a name="licensing"></a>

The data for this project was provided by [Figure Eight / Appen](https://www.figure-eight.com/).