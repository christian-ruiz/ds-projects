'''
AUTHOR: Christian Ruiz
TITLE: Linear Regression Project - Ecommerce
DESCRIPTION: The company is trying to decide whether to focus their efforts on their mobile app experience or their website.
PROJECT START DATE: 07/22/2023
PROJECT END DATE: 
'''

# import packages

# data manipulation
import pandas as pd
import numpy as np

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# upload the data into a df
df = pd.read_csv('linear-regression/Ecommerce Customers.txt')

# review the df
df.head()
df.columns
df.info()
df.describe()

sns.jointplot(data=df, x='Time on Website', y='Yearly Amount Spent')
sns.jointplot(data=df, x='Time on Website', y='Yearly Amount Spent', kind='hex')
sns.jointplot(data=df, x='Time on App', y='Yearly Amount Spent')
# plt.show()

sns.pairplot(df)
# plt.show()

# creating a linear model plot using seaborn
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent', data=df)
plt.show()

'''
the length of the membership is the most correlated with the yearly amount spent
'''

# split between train and testing data
# get columns for the features of the model
df.columns
X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# train the model

lm = LinearRegression()

lm.fit(X_train, y_train)

# reviewing the coefficients of the model
lm.coef_
lm.intercept_

# create df with coefficients to review some of the results
pd.DataFrame(lm.coef_, X.columns, columns=['Coefficients'])

'''
                      Coefficients
Avg. Session Length      25.981550 --- for every unit of avg. session length, yearly amount spent increases by $25.98
Time on App              38.590159 --- for every unit of time on app, yearly amount spent increases by $38.59
Time on Website           0.190405 --- for every unit of time spent on the website, yearly amount spent increases by $0.19
Length of Membership     61.279097 -- for every unit of time for membership length, yearly amount spent increases by $61.28
'''

# use prediction of model on test data
predictions = lm.predict(X_test)

predictions[:6]
y_test.head()

# plot results to compare
sns.scatterplot(y=y_test, x=predictions)
sns.distplot(y_test-predictions)
plt.show()

# calculate model performance with MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error)

print(metrics.mean_absolute_error(y_test, predictions),'\n',
      metrics.mean_squared_error(y_test, predictions), '\n',
      np.sqrt(metrics.mean_squared_error(y_test,predictions)))

# plotting the residuals
sns.distplot(y_test-predictions)
plt.show()

# create a df with the coeff results
pd.DataFrame(lm.coef_, X_train.columns, columns=['Coeff'])

'''
                          Coeff
Avg. Session Length   25.981550 --- for every unit increase in avg. session length, yearly amount spent increases by $25.98
Time on App           38.590159 --- for every unit increase in time on app, yearly amount spent increases by $38.59
Time on Website        0.190405 --- for every unit increase in time on website, yearly amount spent increases by $0.19
Length of Membership  61.279097 --- for every unit increase in length of membership, yearly amount spent increases by $61.28

Conclusion - the company should spend more time on their mobile app
'''

