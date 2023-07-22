
# upload packages
import pandas as pd
import numpy as np
import os
import requests

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# import df
df = pd.read_csv('linear-regression/USA_Housing.csv')
df.head()
df.info()
df.describe()
df.columns

# plot the df
sns.pairplot(df)
plt.show()

# look at the distribution of the price of the houses
sns.displot(df['Price'])
plt.show()

# correlation between columns and heatmap
df.corr()
sns.heatmap(df.corr(),annot=True)
plt.show()

# we're trying to predict price so we will need to split the data between the features and target variables
df.columns

# features
X = df[['Avg. Area Income', 
        'Avg. Area House Age', 
        'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 
       'Area Population'
]]
X.head()

# target variable
y = df['Price']

# splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# setting the linear regression object to a variable
lm = LinearRegression()

# fit the model to the training data
lm.fit(X_train, y_train)

print(lm.intercept_)

# relate to each feature in the dataset
print(lm.coef_)
cdf = pd.DataFrame(lm.coef_, X_train.columns, columns=['Coeff'])
'''
to interperate the data, for each coeff - for each one unit increase, for a specific feature, will result in 
the coeff value increase

                                      Coeff
Avg. Area Income                  21.528276  ---- for every 1 unit increase in income, price increases by $21.53
Avg. Area House Age           164883.282027  ---- for every 1 unit increase in House Age, price increases by $164,883
Avg. Area Number of Rooms     122368.678027  ---- for every 1 unit increase in Number of Rooms, price increases by $122,368
Avg. Area Number of Bedrooms    2233.801864  ---- for every 1 unit increase in Number of Bedrooms, price increases by $2,233
Area Population                   15.150420  ---- for every 1 unit increase in Population, price increases by $15.15
'''
