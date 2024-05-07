#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas_datareader as data  #DATAREADER IS AN API TOREAD FROM NASDAQ,YAHOO ETC
# For reading stock data from yahoo

# For time stamps
from datetime import datetime

# For division
from __future__ import division
from pandas import Series,DataFrame


# In[2]:


from pandas_datareader.data import DataReader
#from pandas_datareader import data


# In[3]:


end=datetime.now()
start=datetime(end.year-1,end.month,end.day)


# In[4]:


#!pip install yfinance


# In[5]:


import yfinance as yf


# In[6]:


data=yf.download('^NSEI',start,end)   #NIFTY50 DATASET
data.head()


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


data.reset_index(inplace=True)
data


# In[ ]:





# In[10]:


data['Adj Close'].plot(figsize=(10,4),legend=True)


# In[11]:


data['Close'].plot(label='close price',legend=True,figsize=(10,4))
data['Open'].plot(label='open price',legend=True)
plt.ylabel("stock price")
plt.title("Nifty open and close price ")
plt.show()



# In[12]:


''''tickers=['TSLA','F','GM']
df=[]
for i in tickers:
    data=yf.download(i,start,end)
    df.append(data)
data.head()'''
data_ford=yf.download('F',start,end)
data_gm=yf.download('GM',start,end)
data_tsla=yf.download('TSLA',start,end)


# In[13]:


data_tsla['Open'].plot(label='TSLA',legend=True)
data_ford['Open'].plot(label='FORD',legend=True)
data_gm['Open'].plot(label='GM',legend=True)
plt.show()


# In[14]:


data_tsla['Volume'].plot(label='TSLA',legend=True)
#d=data_ford['Volume'].max()
data_ford['Volume'].plot(label='FORD',legend=True)
data_gm['Volume'].plot(label='GM',legend=True)
plt.show()


# In[15]:


data_ford.iloc[[data_ford['Volume'].argmax()]]
#data_ford.iloc[[data_ford['Volume'].argmin()]]
#data_ford.iloc[100:200]['Open'].plot(legend=True)
#data_gm.iloc[100:200]['Close'].plot(legend=True)


# In[16]:


data_ford[150:250]['Open'].plot()


# In[17]:


#market capitalization
data_tsla['traded_price']=data_tsla['Open']*data_tsla['Volume']
data_ford['traded_price']=data_ford['Open']*data_ford['Volume']


# In[18]:


data_tsla['traded_price'].plot(label='Tesla', figsize= (15,7))
data_ford['traded_price'].plot(label='Tford', figsize= (15,7))


# In[19]:


data_tsla['traded_price']=data_tsla['Open']*data_tsla['Volume']
data_ford['traded_price']=data_ford['Open']*data_ford['Volume']


# In[20]:


data_tsla['traded_price'].argmax()


# In[21]:


data_ford['traded_price'].argmax()


# In[22]:


data_ford.iloc[[data['Volume'].argmax()]]


# In[23]:


data_tsla.iloc[[data['Volume'].argmax()]]


# In[24]:


#df=data.DataReader('AAPL','yahoo',start,end)


# In[25]:


data.tail()


# In[26]:


data=data.reset_index()  
data.head()


# In[27]:


plt.plot(data.Close)


# In[28]:


#A moving average is a technical indicator that investors and traders use to determine the trend direction of securities. It is calculated by adding up all the data points during a specific period and dividing the sum by the number of time periods. Moving averages help technical traders to generate trading signals.


# # MOVING AVERAGE

# In[29]:


def stock_ma(wks,df):
    ma_close=pd.DataFrame({'Date':df['Date'],'Close':df['Close']})
    ma_close.set_index('Date',inplace=True)
    num=wks*5
    ma_close['movingavg']=ma_close['Close'].rolling(window=num).mean()
    return ma_close.dropna()
stock_ma(5,data).head()


# In[30]:


stock_ma(4,data).plot()


# In[31]:


data_ma_fourweek=stock_ma(5,data)
data_ma_fourweek.reset_index(inplace=True)


# In[35]:


#pip install "altair<5"


# In[ ]:


ma10=data.Close.rolling(10).mean() #moving average
ma200=data.Close.rolling(200).mean()
ma50=data.Close.rolling(50).mean()


# In[ ]:


plt.plot(data.Close)
plt.plot(ma10,'r',label='ma10')
plt.plot(ma200,'b',label='ma20')
plt.plot(ma50,'y',label='ma50')
plt.legend()


# In[ ]:


data.shape


# In[ ]:


train_data=pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
train_data


# In[ ]:


test_data=pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])
test_data


# In[ ]:


from pandas.plotting import scatter_matrix
import pandas as pd
df=pd.concat([data_ford['Open'],data_gm['Open'],data_tsla['Open']],axis=1)
df.columns=['ford open','gm open','tesla open']
scatter_matrix(df,figsize=(8,8))


# In[ ]:


df.corr()


# In[ ]:


ford_reset.columns


# In[ ]:


#pip install mpl_finance


# # Percentage change

# In[37]:


data_tsla['return']=(data_tsla['Close']/data_tsla['Close'].shift(1))-1
data_ford['return']=(data_ford['Close']/data_ford['Close'].shift(1))-1


# In[38]:


data_tsla['return'].plot(label='tesla',kind='kde')
data_ford['return'].hist(bins=50,label='ford')
plt.legend()


# In[39]:


box_data=pd.concat([data_tsla['return'],data_ford['return']],axis=1)
box_data.columns=['Tesla return','ford redurn']
box_data.plot(kind='box')


# In[41]:


#scatter_matrix(box_data,alpha=0.25)


# In[49]:


from datetime import date
end=date(2016,1,1)
start=date(2015,1,1)

data=yf.download('INFY',start,end)
#data

plt.figure(figsize=(17,5))
data.Close.plot()
plt.title("Closing Price",fontsize=20)
plt.show()


# In[51]:


plt.figure(figsize=(17,5))
stock_price = pd.concat([data.Close[:'2015-06-12']/2,data.Close['2015-06-15':]]) # adjustment
plt.plot(stock_price)
plt.title("Closing Price Adjusted",fontsize=20)
plt.show()



# In[56]:


from sklearn.metrics import mean_squared_error as mse
prev_values = stock_price.iloc[:180] #train
y_test = stock_price.iloc[180:] #test

def plot_pred(pred,title):
    plt.figure(figsize=(17,5))
    plt.plot(prev_values,label='Train')
    plt.plot(y_test,label='Actual')
    plt.plot(pred,label='Predicted')
    plt.ylabel("Stock prices")
    plt.title(title,fontsize=20)
    plt.legend()
    plt.show()

y_av = pd.Series(np.repeat(prev_values.mean(),72),index=y_test.index)  #predicted last 72 days
mse(y_av,y_test)

plot_pred(y_av,title="avrg")


# In[57]:


#plot_pred(y_av,title="avrg")

weight = np.array(range(0,180))/180
weighted_train_data =np.multiply(prev_values,weight)

# weighted average is the sum of this weighted train data by the sum of the weight

weighted_average = sum(weighted_train_data)/sum(weight)
y_wa = pd.Series(np.repeat(weighted_average,72),index=y_test.index)

print("MSE: " ,mse(y_wa,y_test))
print("RMSE: " ,np.sqrt(mse(y_wa,y_test)))

plot_pred(y_wa,"Weighted Average")



# In[61]:


y_train = stock_price[80:180]
y_test = stock_price[180:]
print("y train:",y_train.shape,"\ny test:",y_test.shape)

X_train = pd.DataFrame([list(stock_price[i:i+80]) for i in range(100)],
                       columns=range(80,0,-1),index=y_train.index)
X_test = pd.DataFrame([list(stock_price[i:i+80]) for i in range(100,172)],
                       columns=range(80,0,-1),index=y_test.index)

X_train



# In[64]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(X_train,y_train) # Training the models
y_lr = lr.predict(X_test) # inference
y_lr = pd.Series(y_lr,index=y_test.index)

mse(y_test,y_lr), np.sqrt(mse(y_test,y_lr))


# In[65]:


plot_pred(y_lr,"Linear Regression")


# In[ ]:




