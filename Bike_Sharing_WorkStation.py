# Bike Sharing Demand prediction Project Fro the hourly dataset
#
# - We use Knowledge of Basic Python
# - Statistics
# - Data Processing
# - Multiple Linear Reqression 



#-----------------------------------------------
# Step 0 :
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
#-----------------------------------------------

#End

#-----------------------------------------------
# Step 1 - Read the data 
Bikes = pd.read_csv('hour.csv')
#-----------------------------------------------

#End

#-----------------------------------------------
# Step 2 - Prepare Data - Prelim Analysis and Features Selection
Bikes_prep = Bikes.copy() 
Bikes_prep=Bikes_prep.drop(['index','date','casual','registered'],axis=1)
#Bikes_prep.isnull().sum()
# create to Show first impression visualize by pandas histogram
#Bikes_prep.hist(rwidth = 0.9)
#plt.tight_layout()
#-----------------------------------------------

#End

#-----------------------------------------------
# Step 3 - Data visualisation
# Visualise the coninous features Vs demand

# plt.subplot(2,2,1)
# plt.title("Tempereture Vs Demand corr")
# plt.scatter(Bikes_prep['temp'],Bikes_prep['demand'],s=2,c='g')

# plt.subplot(2,2,2)
# plt.title("Etempereture Vs Demand corr")
# plt.scatter(Bikes_prep['atemp'],Bikes_prep['demand'],s=2,c='r')

# plt.subplot(2,2,3)
# plt.title("humidity Vs Demand corr")
# plt.scatter(Bikes_prep['humidity'],Bikes_prep['demand'],s=2,c='m')

# plt.subplot(2,2,4)
# plt.title("windspeed Vs Demand corr")
# plt.scatter(Bikes_prep['windspeed'],Bikes_prep['demand'],s=2,c='b')
# plt.tight_layout()

# # Visualise the Categorical features
# colors = ['g','r','m','b']
# # show season and month avg vs demand by subplot sample visualize
# plt.subplot(2,2,1)
# plt.title('average per demand')
# cat_list = Bikes_prep['season'].unique()
# cat_avg = Bikes_prep.groupby(['season']).mean()['demand']
# plt.bar(cat_list,cat_avg,color=colors)
# # show month avg vs demand
# plt.subplot(2,2,2)
# plt.title('month per demand')
# cat_list = Bikes_prep['month'].unique()
# cat_avg = Bikes_prep.groupby(['month']).mean()['demand']
# plt.bar(cat_list,cat_avg,color=colors)
# plt.tight_layout()

# # show year avg vs demand by indidual plot visualize
# colors = ['g','r','m','b']
# #plt.plot(2,2)
# plt.title('hour average vs demand')
# cat_list = Bikes_prep['hour'].unique()
# cat_avg = Bikes_prep.groupby('hour').mean()['demand']
# plt.bar(cat_list,cat_avg,color=colors)

# Check For Outliers If Exists
#-----------------------------------------------
# To get more focus about where exactly quantile :
#Bikes_prep['demand'].quantile([.05,.15,.5,.75,.95,.99])

#End

#-----------------------------------------------
# Step 4 - Check Muliple Linear Regression Assumption
# Linearity using correlation coffiecient matrix using corr
correlation = Bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr()

#End

#-----------------------------------------------
# Step 5 - drop irrelevant features
Bikes_prep=Bikes_prep.drop(['year','weekday','workingday','atemp','windspeed'],axis=1)
#To Check Autocorrelation of Demand using acor 
df1 = pd.to_numeric(Bikes_prep['demand'],downcast='float')
plt.acorr(df1,maxlags=12)
#-----------------------------------------------

#End

#-----------------------------------------------
# Step 6 - Create/Modify new features like lags
#Log - Normalise the features 'Demand'
df1 = Bikes_prep['demand']
df2=np.log(df1)

# plt.figure()
# df1.hist()

# plt.figure()
# df2.hist()

#Bikes_prep['demand'] = np.log(Bikes_prep['demand'])

t_1 = Bikes_prep['demand'].shift(+1).to_frame()
t_1.columns=['t_1']

t_2 = Bikes_prep['demand'].shift(+2).to_frame()
t_2.columns =['t_2']

t_3 = Bikes_prep['demand'].shift(+3).to_frame()
t_3.columns =['t_3']

Bikes_prep_lags = pd.concat([Bikes_prep,t_1,t_2,t_3],axis=1)

Bikes_prep_lags = Bikes_prep_lags.dropna()
#-----------------------------------------------

#End

# Step 7 - Convert data categorical to dummies 

Bikes_prep_lags['season']=Bikes_prep_lags['season'].astype('category')
Bikes_prep_lags['holiday']=Bikes_prep_lags['holiday'].astype('category')
Bikes_prep_lags['weathersit']=Bikes_prep_lags['weathersit'].astype('category')
Bikes_prep_lags['month']=Bikes_prep_lags['month'].astype('category')
Bikes_prep_lags['hour']=Bikes_prep_lags['hour'].astype('category')


Bikes_prep_lags=pd.get_dummies(Bikes_prep_lags,drop_first=True)
#-----------------------------------------------

#End

#-----------------------------------------------
#Step 8 - Train and Test data - Split
Y=Bikes_prep_lags[['demand']]
X=Bikes_prep_lags.drop(['demand'],axis=1)


# split manullay with out sklearn because we have predict time series 
tr_size=0.7*len(X)
tr_size = int(tr_size)

x_train = X.values[0:tr_size]
x_test =  X.values[tr_size:len(X)]
y_train = Y.values[0:tr_size]
y_test =  Y.values[tr_size:len(Y)]


# 9- Fit and score model : Linear Regression

from sklearn.linear_model import LinearRegression
 
std_reg = LinearRegression()
std_reg.fit(x_train,y_train)

# to calc r square accuracy for model  
r2_train = std_reg.score(x_train,y_train)
r2_test = std_reg.score(x_test,y_test)

# Create Y Predictions 
y_predict = std_reg.predict(x_test)






