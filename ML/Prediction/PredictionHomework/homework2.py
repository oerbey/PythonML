# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

## Data Import
df = pd.read_csv("newSalary.csv")

# Describe Data
#print(df)
#print(df.describe())
#print(df.corr())
#print(df.unvan.unique())

# Visualization of data frame
plt.figure(figsize = (7,5))
salaryDist = sbn.distplot(df["maas"]) # distribution plot.
#plt.savefig("salaryDist.png")

plt.figure(figsize = (7,5))
dfDifForm = sbn.pairplot(df) # it shows all data's in different formats
#plt.savefig("dfDifForm.png")

## SLICING
y = df.iloc[:,-1:].values # get salary column as a numpy
id = df.iloc[:,0:1]
lvl = df.iloc[:,2:-1] # get rest of the columns
x = pd.concat([id, lvl], axis = 1).values
jlvl = df.iloc[:,2:3].values
#print(x)

#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#unvanLE = le.fit_transform(df.iloc[:,1:2])

#n = pd.DataFrame(unvanLE)
#m = pd.concat([n, df.iloc[:,1:2]], axis = 1)
#print(m)

# LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(jlvl,y)

## Visualization of Linear Regression
linPredX = plt.figure(figsize = (10,7))
plt.scatter(jlvl,y, color = 'red')
plt.plot(jlvl,lin_reg.predict(jlvl), color = 'blue')
plt.xlabel("Work Level")
plt.ylabel("Salary")
plt.title("Work Level - Salary Linear Regression")
#plt.savefig("linPredX.png")

# POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # Degree of the polynomial
x_poly = poly_reg.fit_transform(jlvl)
pol_reg = LinearRegression()
pol_reg.fit(x_poly, y)

## Visualization of Polynomial Regression
polyPlt = plt.figure(figsize = (10,7))
plt.scatter(jlvl,y, color = 'green')
# turn into poly values
plt.plot(jlvl, pol_reg.predict(poly_reg.fit_transform(jlvl)), color = 'red')
plt.savefig("polyPlt.png")
