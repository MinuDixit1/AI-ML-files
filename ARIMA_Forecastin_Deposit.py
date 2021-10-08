"""
Created on Sat Oct 31 21:01:21 2020

@author: himanshukoli
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot

from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.statespace.sarimax import SARIMAX
Excel_Name = 'ac_statement.xlsx'
Sheet_Name = 'DS_1'

df1 = timeseries_deposit(Excel_Name, Sheet_Name)

df1.plot()


df2 = df1.reset_index()


ts= df1.Deposit_Amount

df2 = df1.reset_index()

plt.plot(ts)

ts_week = ts.resample('W').mean()
ts_week[:5]

plt.plot(ts_week)


def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window = 52).mean() #rolling mean
    rolstd = timeseries.rolling(window = 52).std()
    org = plt.plot(timeseries, color = 'blue',label = 'original')
    meangraph = plt.plot(rolmean, color = 'red',label = 'Rolling mean')
    stdgraph = plt.plot(rolstd, color = 'black',label = 'Rolling std')
    plt.legend(loc='best')
    plt.show(block=False)
    dftest = adfuller(timeseries)
    print(dftest)
    print("Test statstics",dftest[0])
    print("p value",dftest[1])
    print("#lags used ", dftest[2])
    
test_stationarity(ts)

ts_week_log =  np.log1p(ts)

test_stationarity(ts_week_log)

plt.plot(ts_week_log)

ts_week_log_diff = ts_week_log - ts_week_log.shift()
ts_week_log_diff 
ts_week_log_diff.dropna(inplace=True)
test_stationarity(ts_week_log_diff)

lag2 = ts_week_log_diff-ts_week_log_diff.shift()

lag2.dropna(inplace=True)
test_stationarity(lag2)

df2['Deposit_Amount'] = np.log1p(df2['Deposit_Amount'] )

#acf = p
#pacf = q

df2['is_holiday'] = df2['Deposit_Amount'].apply(lambda x: 1 if x == 0  else 0)

train=df2[(df2['Date'] >= '2015-01-01' ) & (df2['Date'] <= '2018-01-01' )]

test=df2[(df2['Date'] > '2018-01-01')]



autocorrelation_plot(df2['Deposit_Amount'])
pyplot.show()

lag_acf = acf(train['Deposit_Amount'],nlags = 30) 

lag_pcf = pcf(ts_week_log_diff, nlags = 10, method = '0ls')

plt.plot(lag_acf)
plt.plot(lag_pcf)



plot_acf(ts_week_log_diff,lags=25)
pyplot.show()


model_ar = ARIMA(train['Deposit_Amount'], order=(20,1,0))
model_fit = model_ar.fit(disp=0)

print(model_fit.summary())

model_fit.plot_diagnostics()
plt.show()

train.set_index('Date')

result = seasonal_decompose(train.set_index('Date'),model='additive',period=7)
result.plot()
pyplot.show()

seaonal_model = SARIMAX(train['Deposit_Amount'], order = (13,1,0), seaonal_order = (13,1,0,7))

seaonal_fit = seaonal_model.fit()

print(seaonal_fit.summary())

seaonal_fit.plot_diagnostics()

aa = seaonal_fit.forecast()

forecast = seaonal_fit.get_prediction(start = 793, end = 1525, dynamic=False)
predictions = forecast.predicted_mean


history = [x for x in train['Deposit_Amount']]
test_p =  [x for x in test['Deposit_Amount']] 
predictions = list()
for t in range(len(test_p)):
    seaonal_model = SARIMAX(history,order = (20,1,0), seaonal_order = (20,1,0,7))
    seaonal_fit = seaonal_model.fit()
    yhat = seaonal_fit.forecast()
    predictions.append(yhat)
    obs = test_p[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test_p, predictions)
print('Test MSE: %.3f' % error)



myoupt = model_fit.predict(typ = 'levels',start = 1, end= 1000)

myoutput = model_fit.forecast(steps=428)

ouputlist = myoutput[0]




final_test = np.expm1(ouputlist)

print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

history = [x for x in train['Deposit_Amount']]
test_p =  [x for x in test['Deposit_Amount']] 
predictions = list()
for t in range(len(test_p)):
	model = ARIMA(history, order=(10,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_p[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test_p, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test_p)
pyplot.plot(predictions, color='red')
pyplot.show()


final_test = np.expm1(predictions)

error = mean_squared_error(test_p, final_test)
print('Test MSE: %.3f' % error)

mydf = df1.reset_index()
    
mytest = mydf[(mydf['Date'] > '2018-01-01')]

test_actual = [x for x in mytest['Deposit_Amount']]

error = math.sqrt(mean_squared_error(test_actual, final_test))
print('Test MSE: %.3f' % error)

pyplot.plot(final_test)
pyplot.plot(test_actual)

import math
    
    
