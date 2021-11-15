
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File & Folder Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run with no issues using Python versions 3.*.

Beyond the Anaconda distribution of Python, a library called "plotly" has been used for visualisations.

To install plotly, type `pip install plotly` into your command line interface.

For more detailed instructions, see https://pypi.org/project/plotly/

## Project Motivation<a name="motivation"></a>

For this project, I am using publicly available datasets to understand which countries would have the best chance of making a Developer happy. 

Three questions were asked in the pursuit of answers:

1. Which countries lead to the best possible combination of high job satisfaction and salary?
2. Which countries are the happiest in the world?
3. Given a certain values profile for a Developer, which countries maximize happiness?

Three sources of data were used in order to answer the above questions:

1. [Stack Overflow Developer Survey 2020](https://insights.stackoverflow.com/survey/), which contains data on Developer salaries and job satisfaction?
2. [World Happiness Ratings 2020](https://www.kaggle.com/mathurinache/world-happiness-report/version/1?select=2020.csv), which contains data on the countries and their ladder score.
3. [Gap Minder World Population Data 2007](https://www.gapminder.org/tag/population-data/), only used for quickly getting ISO codes for countries, for the purpose of doing geographic plots (note that this data is directly accessed through plotly, see Jupyter notebook.)


## File & Folder Descriptions <a name="files"></a>

There is 1 notebook called "Developer Seeking Happiness – The Planetary Picture" which showcases the work related to answering the above questions. Comments and markdown have been used in the notebook to help clarify the steps taken in the code and note some observations. 

Note that the [static render of plotly images](https://github.com/plotly/plotly.py/issues/931) means no interactive plots are displayed on github - it is best to download the repository and run it locally, or view results in the [Medium blog post](https://medium.com/@s21a/developer-seeking-happiness-a-planetary-perspective-61f4af944c3c).

There are three folders in the repository as described below:

1. "kaggle_wold_happiness_2020_survey" contains the csv from [World Happiness Ratings 2020](https://www.kaggle.com/mathurinache/world-happiness-report/version/1?select=2020.csv).
2. "stackoverflow_2020_survey" contains the unzipped data from [Stack Overflow Developer Survey 2020](https://insights.stackoverflow.com/survey/).
3. "blog_post" contains the screenshots and gifs used for the [Medium blog post](https://medium.com/@s21a/developer-seeking-happiness-a-planetary-perspective-61f4af944c3c).

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@s21a/developer-seeking-happiness-a-planetary-perspective-61f4af944c3c).

## Acknowledgements<a name="licensing"></a>

I did not create this data, only downloaded and fused it to generate insights. Credit for the sources of data are detailed below. Otherwise, feel free to used the code here as you wish.

Credit for Developer Survey: 

* Stack Overflow.

Credit for World Happiness Data: 

* Helliwell, John F., Richard Layard, Jeffrey Sachs, and Jan-Emmanuel De Neve, eds. 2020. World Happiness Report 2020. New York: Sustainable Development Solutions Network
* Mathurin Aché https://www.kaggle.com/mathurinache

