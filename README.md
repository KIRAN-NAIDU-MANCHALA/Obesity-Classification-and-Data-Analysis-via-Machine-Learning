#### Obesity-Classification-and-Data-Analysis-via-Machine-Learning
### Introduction
# I chose to further analyze the dataset I used for my EDA for the final project. The dataset recorded the obesity levels of people from Mexico, Peru, and Colombia alongside their eating habits and physical condition. As the project asked us to build a machine learning model, I was interested in building an accurate model around if a person is obese or not — a two-class problem — as well as finding the features that would be most relevant in training this model.


#The dataset I used has the data of 2111 individuals aged 14 to 61 and 17 attributes. Many of these attributes have acronyms, so I briefly described all of them below:

 - Gender: 1= female, 2 = male
 - Age: numeric
 - Height: numeric, in meters
 - Weight: numeric, in kilograms
 - family_history (family history of obesity): 1 = yes, 2 = no
 - FCHCF (frequent consumption of high caloric food): 1= yes, 2= no
 - FCV (frequency of consumption of vegetables: 1 = never, 2 = sometimes, 3 = always
 - NMM (number of main meals): 1, 2, 3 or 4 meals a day
 - CFBM (consumption of food between meals): 1=no, 2=sometimes, 3=frequently, 4=always
 - Smoke: 1= yes, 2= no
 - CW (consumption of water): 1 = less than a liter, 2 = 1–2 liters, 3 = more than 2 liters
 - CCM (calorie consumption monitoring): 1= yes, 2 = no
 - PAF (physical activity frequency per week): 0 = none, 1 = 1 to 2 days, 2= 2 to 4 days, 3 = 4 to 5 days
 - TUT (time using technology devices a day): 0 = 0–2 hours, 1 = 3–5 hours, 2 = more than 5 hours
 - CA (consumption of alcohol): 1= never, 2 = sometimes, 3 = frequently, 4 = always
 - Transportation: 1 = automobile, 2 = motorbike, 3 = bike, 4 = public transportation, 5= walking
 - Obesity (target variable): 2 = not obese, 4 = obese

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
