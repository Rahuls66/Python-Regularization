import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import ElasticNet

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train = df_train.dropna()
df_test = df_test.dropna()

x_train = df_train['x']
x_train = x_train.values.reshape(-1,1)

y_train = df_train['y']
y_train = y_train.values.reshape(-1,1)


x_test = df_test['x']
x_test = x_test.values.reshape(-1,1)

y_test = df_test['y']
y_test = y_test.values.reshape(-1,1)

enet = ElasticNet()

enet.fit(x_train, y_train)

print("Ridge Train RMSE:", np.round(np.sqrt(metrics.mean_squared_error(y_train, enet.predict(y_train))),5))
print("Ridge Test RMSE:", np.round(np.sqrt(metrics.mean_squared_error(y_test, enet.predict(y_test))),5))
