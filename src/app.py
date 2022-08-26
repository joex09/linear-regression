#Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import statsmodels.api as sm
from utils import Helpers

#Load and Read 
df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')

#Utilización de utils.py
helpers  = Helpers()
helpers.conv_region(region_name='region')

#Vectorizar   
df['region'] = df.apply(lambda x: conv_region(x['region']), axis=1)

#Recategorizacion
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True,cmap='viridis', vmax=1, vmin=-1, center=0)

#Split
X = df.drop(['charges'], axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

#Results
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

#Examples
edad = 22
sex = 1
bm = 21
children = 0
smoker = 0
region = 3

print('Predicted prima : \n', regr.predict([[edad,sex,bm,children,smoker,region]]))

#Another model
x = np.array(df[["age", "sex", "bmi", "smoker"]])
y = np.array(df["charges"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

forest = RandomForestRegressor()
forest.fit(xtrain, ytrain)

ypred = forest.predict(xtest)
df= pd.DataFrame(data={"Predicted Premium Amount": ypred})
print(df.head())