# Trying to predict sales by months

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
df = pd.read_csv("satislar.csv")

#print(df.head())

months = df[["Aylar"]]
sales = df[["Satislar"]] # Sales2 = df.iloc[:,:1].values

# Data split for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size = 0.33, random_state = 42)
'''
# data scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''
# Build linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

prediction = lr.predict(x_test,)
# print(prediction)

# Visualization
x_train = x_train.sort_index()
y_train = y_train.sort_index()
salesFig = plt.plot(x_train, y_train)
#plt.savefig("salesFig.png")

# draw closest line to our data for lineer regression
predictFig = plt.plot(x_test, lr.predict(x_test))
plt.title("Sales by Months")
plt.xlabel("Months")
plt.ylabel("Months")
#plt.savefig("predictFig.png")
