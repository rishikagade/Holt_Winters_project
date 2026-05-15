#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import datetime


# In[2]:


Daily = pd.read_csv('Gold_Daily .csv')
print(Daily.head(5))
print('------------------')
print(Daily.dtypes)
print('------------------')
print(Daily.describe())
print('------------------')
print(Daily.apply(lambda x: pd.isna(x).sum(), axis = 0))


# In[3]:


#Set the date to a 'datetime' data type and set it as the index
Daily['Date'] = pd.to_datetime(Daily['Date'])
Daily = Daily.set_index('Date')


# In[4]:


Daily['PriceMovement'] = ['UP' if Price>=Open else 'Down' for Price,Open in zip(Daily['Price'],Daily['Open'])]
colormap = {'UP':'green', 'Down':'red'}
Daily['BarHeight'] = [Price - Open if Price>=Open else Open - Price for Price,Open in zip(Daily['Price'],Daily['Open'])]
Daily['BarStart'] = [Price if Open > Price else Open for Price,Open in zip(Daily['Price'],Daily['Open'])]


# In[8]:


fig,ax = plt.subplots(figsize = (15,10))
ax.bar(Daily.index, bottom = Daily['BarStart'],height = Daily['BarHeight'], color = Daily['PriceMovement'].map(colormap))
ax.bar(Daily.index, bottom = Daily['Low'],height = Daily['High']-Daily['Low'],color = Daily['PriceMovement'].map(colormap), width = 0.25)
plt.title('Candlestick plot of Daily Price')
plt.xlabel('Date')
plt.ylabel('USD')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
plt.show()


fig,ax = plt.subplots(figsize = (10,7))
ax.plot(Daily.index,Daily['Price'])
plt.title('Line plot of Closing Price')
plt.xlabel('Date')
plt.ylabel('USD')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
plt.show()


# In[9]:


#Look at seasonal decomposition to see if there's a seasonality component here
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(Daily['Price'])
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose 
decompose_result = seasonal_decompose(Daily['Price'],period=1)
decompose_result.plot()
plt.show()


# In[10]:


#importing inbuilt library for holt winters'
from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_model = ExponentialSmoothing(Daily['Price'],trend='add').fit()

test_predictions = fitted_model.forecast(90).rename('HW Test Forecast for next 90 days')
print(test_predictions[:-10])
idx = pd.date_range(max(Daily.index), periods=90, freq="D")
test_predictions.index = idx


Daily['Price'].plot(legend=True,label='Historic Daily Close Price')
test_predictions.plot(legend=True,label='Predicted Close Price',figsize = (10,7),xlim=[max(Daily[:-10].index),max(test_predictions.index)])
plt.title('Predicted Price using Holts Winters Double Exponential smoothing');


# In[ ]:


mu = Daily['Price'][:-90].pct_change().mean()
sigma = Daily['Price'][:-90].pct_change().std()
simulation = {}
simulation["Date"] = pd.date_range(start=Daily.index[-1],end=Daily.index[-1] +datetime.timedelta(days=90)) 
for sim in range(1,10000): 
    StartValue = Daily['Price'].iloc[-1] 
    simulation["Simulation_"+str(sim)] = [] 
    simulation["Simulation_"+str(sim)].append(StartValue) 
    for days in range(90): 
        next_day = simulation["Simulation_"+str(sim)][-1]*np.exp((mu-(sigma**2/2))+sigma*np.random.normal()) 
        simulation["Simulation_"+str(sim)].append(next_day)


simulation = pd.DataFrame(simulation) 
ax_base = simulation.set_index('Date').plot(legend=False,title = 'Simulated Price Walk for Gold Price\nNext 90 days') 
vals = ax_base.get_yticks() 
ax_base.set_yticklabels(['${:,.2f}'.format(x) for x in vals]) 
plt.xlabel('Date') 
plt.ylabel('Price') 
plt.show() 


# In[ ]:


print('An analysis of the final results of the simulations')
print(f"The average ending price of all simulations is {'${:,.2f}'.format(simulation.iloc[-1,1:].mean())}") 
print(f"The variation of the ending price for all simulations is {'${:,.2f}'.format(simulation.iloc[-1,1:].std())}") 
print(f"95% range for all simulations between {'${:,.2f}'.format(np.quantile(simulation.iloc[-1,1:],0.025))} and {'${:.2f}'.format(np.quantile(simulation.iloc[-1,1:],0.975))}") 



plot2 = plt.figure(figsize = (10,7)) 
ax1 = simulation.iloc[-1,1:].plot.hist(bins = 30, alpha = 0.5,title = 'Histogram of simulation results') 
vals = ax1.get_xticks() 
ax1.set_xticklabels(['${:,.2f}'.format(x) for x in vals])
plt.show()


# In[ ]:




