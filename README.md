# Obesity-Classification-and-Data-Analysis-via-Machine-Learning

# Abstract

#### The epidemic of overweight and obesity presents a major challenge to chronic 
#### disease prevention and health across the life course around the world. 
#### Fueled by economic growth, industrialization, mechanized transport, urbanization,
#### an increasingly sedentary lifestyle, and a nutritional transition to processed 
#### foods and high calorie diets over the last 30 years, many countries have witnessed
#### the prevalence of obesity in its citizens double, and even quadruple. Rising prevalence
#### of childhood obesity,in particular, forebodes a staggering burden of disease in individuals
#### and healthcare systems in the decades to come. A complex, multifactorial disease, with genetic,
#### behavioral, socioeconomic,and environmental origins, obesity raises risk of debilitating morbidity
#### and mortality. Relying primarily on epidemiologic evidence published within the last decade, this 
#### non-exhaustive review discusses the extentnof the obesity epidemic, its risk factors—known 
#### and novel—, sequelae, and economic impact across the globe.
## Introduction
#### I chose to further analyze the dataset I used for my EDA for the final project. The dataset recorded the obesity levels of people from Mexico, Peru, and Colombia alongside their eating habits and physical condition. As the project asked us to build a machine learning model, I was interested in building an accurate model around if a person is obese or not — a two-class problem — as well as finding the features that would be most relevant in training this model.


 - The dataset I used has the data of 2111 individuals aged 14 to 61 and 17 attributes. Many of these attributes have acronyms, so I briefly described all of them below:

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
 - Obesity (target variable):1 = Normal_Weight, 2 = Obesity_Type_I  , 3 = The person is Obesity_Type_II , 4 = The person is Obesity_Type_III , 5 = The person is Overweight_Level_I , 6 = The person is Overweight_Level_II   .

# Data Preparation
## 1.Import libraries


 - %matplotlib inline
 - import matplotlib.pyplot as plt
 - import numpy as np
 - import pandas as pd
 - from sklearn import linear_model
 - from sklearn.metrics import mean_squared_error, r2_score
 - from sklearn.preprocessing import PolynomialFeatures

# IMPORTING DATA SET

 - df = pd.read_csv(r"C:\Users\Hi\Desktop\ObesityDataSet_raw_and_data_sinthetic.csv")
 - df.head()

![1](https://user-images.githubusercontent.com/92929087/138436411-dab2ede5-c435-4192-b9a7-dc4ca84074df.png)

## First, I imported the libraries I would need to understand and train my data. After, I used google.colab to import the CSV file, and then loaded the data into a      data frame using pandas. The first five data points are shown through the .head function.

# CHECKING SHAPE AND VALUSES

 - df.shape
![2](https://user-images.githubusercontent.com/92929087/138436991-3873a774-e009-4836-a6f2-55e5ea9239b7.png)
 
 - df.dtypes

![3](https://user-images.githubusercontent.com/92929087/138436998-e1a15bb3-f3fb-4422-a021-255036aa4f34.png)

 - df.isnull().sum()

![6](https://user-images.githubusercontent.com/92929087/138437492-28f31e16-2681-4e38-9744-1e03e340ead8.png)


## The .shape function accurately returned (2111,17), and the heatmap confirmed that there are no missing values.


# Premodeling (CHANGING ALL TERMS TO NUMERIC BY USING THE LABEL ENCODER)
### 1.Preprocessing

 - from sklearn.preprocessing import LabelEncoder

 - le = LabelEncoder()

 - for col in df.columns[~(df.columns.isin(['Age']))].tolist():
   -  df[col] = le.fit_transform(df[col])
   -  df

![image](https://user-images.githubusercontent.com/92929087/138437744-86561630-ab62-4cb2-982d-eda65306448d.png)

   
 #### Before the data can be split, it should be normalized because the ranges of the dataset features are not the same. This can be problematic because a small change in a feature may not affect the other, so the ranges are normalized to a uniform range of 0–1.
  
 # 2. Splitting test and training data 
 - X=df[['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS','NObeyesdad']]
 - y = df['NObeyesdad']
 - X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,random_state=100)
    
 - from sklearn.model_selection import train_test_split

 - X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

#For train data
y_prob_train = log_reg.predict_proba(X_train)[:,1]
y_pred_train = log_reg.predict(X_train)


# for test data
y_prob = log_reg.predict_proba(X_test)[:,1]
y_pred = log_reg.predict(X_test) #to change threshold should do it manually







 # Logistic Regression
### Logistic Regression
### One of the basic linear models developed with a probabilistic approach to classification problems is Logistic Regression (31) and is one of the supervised
###  learning models widely used in ML. Logistic Regression can be seen as a development of Linear Regression models with a logistic function for data with a
### target in the form of classes (32) as follows:

### y(x)=σ(β0+βTx ),
### where x=(x1,x2,…,xD)T is the D-dimensional data, β=(β1,β2,…,βD)T are the weight parameters, β0 is the bias parameter, and σ is a logistic function that is shaped ### as σ(a)=11+e−a.

### The weights of β can be obtained by using probabilistic concepts. For example, if yn = y(xn) and tn ∈ {0, 1} are an independent identical distribution. The joint probabilistic or likelihood function for all the data can be expressed by the Bernoulli distribution p(t|β), where t=(t1,t2,…,tN)T. Therefore, the Logistic Regression learning and bias (β) is to maximize p(t∨β). The learning method for determining the weight and bias (β) parameters is known as the maximum likelihood method. Generally, the solution to the maximum likelihood problem is done by minimizing the negative of the logarithm of the likelihood function, namely minβ E(β), where E(β) = −ln(p(t∨β)). Logistic Regression models can use regularization techniques to solve the problem of overfitting by adding the weight norm ||w|| in the error function, namely E(β)=12∣∣∣∣β∣∣∣∣2+C∑Nn=1{tnln(yn)+(1−tn)ln(1−yn)}, where C > 0 is the inverse parameter of the regulation.

### Simultaneous and partial parameter testing is performed to examine the role of predictor variables in the model. Simultaneous parameter testing uses the G test.

 - from sklearn.linear_model import LogisticRegression
 - from sklearn.model_selection import train_test_split

 - X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=31)



 - from sklearn.linear_model import LogisticRegression

 - log_reg = LogisticRegression(solver='liblinear', fit_intercept=True, random_state=1)

 - log_reg.fit(X_train, y_train)

![99](https://user-images.githubusercontent.com/92929087/138446650-40162eff-ff88-41bb-b96a-fd417dc31a5a.png)




 
 
 
 # Accuracy



from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print('Accuracy for test: ',accuracy_score(y_test,y_pred))

![image](https://user-images.githubusercontent.com/92929087/138447208-b770b912-6321-4a8f-8342-4124d229424e.png)


 # ACCURACY FOR TRAIN DATA SET
  - print('accuracy - Train: ',accuracy_score(y_train,y_pred_train))



# ACCURCY FOR TEST DATASET
 - print('accuracy - Test: ',accuracy_score(y_test,y_pred))




# confusion matrix
 - print('Confussion matrix : ','\n',confusion_matrix(y_test,y_pred))

![image](https://user-images.githubusercontent.com/92929087/138447320-d1c244bf-f5a6-4992-a62b-60d18e9afae4.png)




 # RECALL ,PRECISION,F1-SCORE
 
 ## FOR TRAIN DATASET
  - print(classification_report(y_train,y_pred_train))


## FOR TEST DATASET
 - print(classification_report(y_test,y_pred))
 - ![t2](https://user-images.githubusercontent.com/92929087/140480124-50c0bfb3-f9a7-4f4b-95da-99c2766ed3df.PNG)





# mean squared error of the model
 - mse = mean_squared_error(y_test,y_pred)
 - mse

![image](https://user-images.githubusercontent.com/92929087/138447594-5ed7a815-5d86-4458-8a3d-2d34ba53e4ec.png)




# Finding error.
 - rmse =np.sqrt(mean_squared_error(y_test,y_pred))
 - rmse

![image](https://user-images.githubusercontent.com/92929087/138447671-67aa70bd-3cfe-4dfc-8906-b2157afacf78.png)




 - import seaborn as sns
 - sns.distplot(df.NObeyesdad)
 - plt.show()

![image](https://user-images.githubusercontent.com/92929087/138447889-72d2f4a9-78b0-41e2-9d9d-84cac9244500.png)


## Testing model
 - scaler = StandardScaler()
 - scaler.fit(X)
 - standardized_data = scaler.transform(X)
 - classifier = svm.SVC(kernel='linear')
 - training the support vector Machine Classifier
 - classifier.fit(X_train, y_train)
 - input_data = (0,21.000000,295,245,1,0,170,477,2,0,549,0,0,840,3,3)


### changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

### reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

### standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 1):
  print('The person is Normal_Weight')
elif(prediction[0] == 5):
  print('The person is Overweight_Level_I')
elif(prediction[0] == 6):
  print('The person is Overweight_Level_II')
elif(prediction[0] == 4):
  print('The person is Obesity_Type_III')
elif(prediction[0] == 0):
  print('The person is Insufficient_Weight')
elif(prediction[0] == 2):
  print('The person is Obesity_Type_I')
elif(prediction[0] == 3):
  print('The person is Obesity_Type_II')
  
![1](https://user-images.githubusercontent.com/92929087/140477399-5a220cfb-d1ef-4b0c-914f-6333806cbe58.PNG)


















![This is an image](https://myoctocat.com/assets/images/base-octocat.svg)
