import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

customers = pd.read_csv('Ecommerce Customers')
print(customers.columns)
# sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind='hex')
# sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
# plt.show()

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)
# print(lm.coef_)
res = lm.predict(X_test)
print(res)
# sns.scatterplot(y_test, res)
sns.distplot((y_test-res),bins=50);
plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, res))
print('MSE:', metrics.mean_squared_error(y_test, res))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, res)))

coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
