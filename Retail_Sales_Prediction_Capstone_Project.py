#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/asim5800/Retail-Sales-Prediction/blob/main/Retail_Sales_Prediction_Capstone_Project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # <b><u> Project Title : Sales Prediction : Predicting sales of a major store chain Rossmann</u></b>

# ## <b> Problem Description </b>
# 
# ### Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.
# 
# ### You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.

# ## <b> Data Description </b>
# 
# ### <b>Rossmann Stores Data.csv </b> - historical data including Sales
# ### <b>store.csv </b> - supplemental information about the stores
# 
# 
# ### <b><u>Data fields</u></b>
# ### Most of the fields are self-explanatory. The following are descriptions for those that aren't.
# 
# * #### Id - an Id that represents a (Store, Date) duple within the test set
# * #### Store - a unique Id for each store
# * #### Sales - the turnover for any given day (this is what you are predicting)
# * #### Customers - the number of customers on a given day
# * #### Open - an indicator for whether the store was open: 0 = closed, 1 = open
# * #### StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
# * #### SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
# * #### StoreType - differentiates between 4 different store models: a, b, c, d
# * #### Assortment - describes an assortment level: a = basic, b = extra, c = extended
# * #### CompetitionDistance - distance in meters to the nearest competitor store
# * #### CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
# * #### Promo - indicates whether a store is running a promo on that day
# * #### Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
# * #### Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
# * #### PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

# ## Loading Libraries & Data

# In[1]:


get_ipython().system('pip3 install -r requirements.txt')


# In[2]:


get_ipython().system('pip install python-dotenv')


# In[3]:


from dbrepo.RestClient import RestClient
import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

client = RestClient(endpoint="https://test.dbrepo.tuwien.ac.at", username=os.getenv("DBREPO_USERNAME"), password=os.getenv("DBREPO_PASSWORD"))
store = client.get_identifier_data(identifier_id="9627ec46-4ee6-4969-b14a-bda555fe34db") 
store

#store https://test.dbrepo.tuwien.ac.at/pid/9627ec46-4ee6-4969-b14a-bda555fe34db https://handle.test.datacite.org/10.82556/nqeg-gy34
#train https://test.dbrepo.tuwien.ac.at/pid/b1c59499-9c6e-42c2-af8f-840181e809db https://handle.test.datacite.org/10.82556/yb6j-jw41
#test https://test.dbrepo.tuwien.ac.at/pid/7cbb845c-21dd-4b60-b990-afa8754a0dd9 https://handle.test.datacite.org/10.82556/jerg-4b84


# In[4]:


import pandas as pd
data_chunks = []

page = 0
size = 10000
while True:
    chunk = client.get_table_data(
        database_id="18021ccb-88bd-41af-98db-835cb7dc7354",
        table_id="d81e3014-4ad0-4ea5-91df-8b1e90e87ff7",
        page=page,
        size=size
    )
    
    if chunk.empty:
        break
    data_chunks.append(chunk)
    page += 1

store_train = pd.concat(data_chunks, ignore_index=True)
store_train.head()


# In[5]:


store_train.info()


# In[6]:


data_chunks = []

page = 0
size = 10000 
while True:
    chunk = client.get_table_data(
        database_id="18021ccb-88bd-41af-98db-835cb7dc7354",
        table_id="17e50f8e-b94b-407d-97fc-abd9b9cb422e",
        page=page,
        size=size
    )
    
    if chunk.empty:
        break 
    data_chunks.append(chunk)
    page += 1 

store_test = pd.concat(data_chunks, ignore_index=True)
store_test.info()


# In[7]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import matplotlib
import matplotlib.pylab as pylab

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 8,6

get_ipython().run_line_magic('pip', 'install statsmodels')

import math
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LassoLars
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet


# ## **Analysing the Rossman Dataset**

# In[8]:


store_train.head()


# In[9]:


store_train.tail()


# ##### **Checking Information about Dataset**

# In[10]:


store_train.shape


# In[11]:


#Checking info of data as data types and rows and cols
store_train.info()


# In[12]:


#Checking Null Values
store_train.isnull().sum()


# **Summary Statastics Of Dataset**

# In[13]:


#Summary Statastics
store_train.describe()


# In[14]:


#No. Of Stores in the Dataset
store_train.store.nunique()


# In[15]:


# Value_counts of StateHoliday Column
store_train['stateholiday'].value_counts()


# In[16]:


print(store_train.date.min(),'initial')
print(store_train.date.max(),'final')


# **This tells us we have a data of almost 3 years.**

# In[17]:


# extract year, month, day and week of year from "Date"

store_train['date']=pd.to_datetime(store_train['date'])
store_train['year'] = store_train['date'].apply(lambda x: x.year)
store_train['month'] = store_train['date'].apply(lambda x: x.month)
store_train['day'] = store_train['date'].apply(lambda x: x.day)
store_train['weekofyear'] = store_train['date'].apply(lambda x: x.weekofyear)


# In[18]:


store_train.sort_values(by=['date','store'],inplace=True,ascending=[False,True])
store_train


# ## **EDA On Rossman Dataset**

# #### **Heatmap of the Rossman Dataset**

# In[19]:


store_train['open'] = store_train['open'].map({'true': 1, 'false': 0})
store_train['promo'] = store_train['promo'].map({'true': 1, 'false': 0})
store_train = store_train.drop(['date'], axis=1)
numeric_features = store_train.select_dtypes(include=[np.number])
correlation_map = numeric_features.corr()
obj = np.array(correlation_map)
obj[np.tril_indices_from(obj)] = False
fig, ax = plt.subplots()
fig.set_size_inches(9,9)
sns.heatmap(correlation_map, mask=obj, vmax=.7, square=True, annot=True)
plt.show()


# **As we can see that in the graph given below that Stores mainly closed on Sunday**

# In[20]:


sns.countplot(x='dayofweek',hue='open',data=store_train)


# **Sales Are nearly doubled High When Promo is Running**

# In[21]:


#Impact of promo on sales
store_train['sales'] = pd.to_numeric(store_train['sales'], errors='coerce')

Promo_sales = pd.DataFrame(store_train.groupby('promo').agg({'sales':'mean'}))

sns.barplot(x=Promo_sales.index, y=Promo_sales['sales'])
plt.xlabel('Promo')
plt.ylabel('Average Sales')
plt.title('Average Sales by Promo')
plt.show()


# **As We can see that In the month of November and Specially in December Sales is increasing Rapidly every year on the christmas eve.**

# In[22]:


sns.catplot(x="month", y="sales", data=store_train, kind="point", aspect=2, height=10)

plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales by Month')
plt.show()


# In[23]:


# Value Counts of SchoolHoliday Column
store_train.schoolholiday.value_counts()


# **As we can see in the Piechart Sales affected by School Holiday is 18% and Mainly Sales aren't afffected by School Holiday**

# In[24]:


labels = 'Not-Affected' , 'Affected'
sizes = store_train.schoolholiday.value_counts()
colors = ['gold', 'silver']
explode = (0.1, 0.0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=180)
plt.axis('equal')
plt.title("Sales Affected by Schoolholiday or Not ?",fontsize=20)
plt.plot()
fig=plt.gcf()
fig.set_size_inches(6,6)
plt.show()


# ### **Transforming Variable StateHoliday**

# **As we can see in the Piechart Sales affected by State Holiday is only 3% means Sales aren't afffected by State Holiday**

# In[25]:


sizes = store_train['stateholiday'].value_counts()
labels = sizes.index
colors = sns.color_palette('pastel')[0:len(labels)]
explode = [0.1] + [0 for _ in range(len(labels)-1)]

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, colors=colors,
       autopct='%1.1f%%', shadow=True, startangle=180)
ax.axis('equal')
plt.title("State Holiday Types", fontsize=20)
fig.set_size_inches(6,6)
plt.show()


# **As Sales isn't much affected by State Holiday so i'm removing this column**

# In[26]:


store_train.drop('stateholiday',inplace=True,axis=1)


# **Histogram Representation of Sales. Here 0 is showing because most of the time store was closed.**

# In[27]:


#distribution of sales
fig, ax = plt.subplots()
fig.set_size_inches(11, 7)
sns.distplot(store_train['sales'], kde = False,bins=40);


# **Sales vs Customers**

# In[28]:


store_train['sales'] = pd.to_numeric(store_train['sales'], errors='coerce')
store_train['customers'] = pd.to_numeric(store_train['customers'], errors='coerce')

sns.lmplot(x='sales', y='customers', data=store_train, palette='seismic', height=5, aspect=1, line_kws={'color':'blue'})

plt.title('Linear Relation Between Sales and Customers', fontsize=15)
plt.show()


# ## **Analysing the Store Dataset**

# In[29]:


store.head(5)


# In[30]:


store.tail()


# ##### **Checking Information about Dataset**

# In[31]:


store.shape


# In[32]:


#Checking info of data as data types and rows and cols
store.info()


# In[33]:


#Checking Null Values
store.isnull().sum()


# **Heatmap for null values**

# In[34]:


# creating heatmap for null values
plt.figure(figsize=(10,6))
sns.heatmap(store.isnull(),yticklabels= False, cbar= False, cmap= 'gnuplot')


# **Distribution Of Different Store Types**

# In[35]:


labels = 'a' , 'b' , 'c' , 'd'
sizes = store.storetype.value_counts()
colors = ['orange', 'green' , 'red' , 'pink']
explode = (0.1, 0.0 , 0.15 , 0.0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=180)
plt.axis('equal')
plt.title("Distribution of different StoreTypes")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(6,6)
plt.show()


# ### **Remove features with high percentages of missing values**
# 
# #### **we can see that some features have a high percentage of missing values and they won't be accurate as indicators, so we will remove features with more than 30% missing values.**

# In[36]:


# remove features
store = store.drop(['competitionopensincemonth', 'competitionopensinceyear','promo2sinceweek',
                     'promo2sinceyear', 'promointerval'], axis=1)


# #### **Replace missing values in features with low percentages of missing values**

# In[37]:


# CompetitionDistance is distance in meters to the nearest competitor store
# let's first have a look at its distribution

sns.distplot(store.competitiondistance.dropna())
plt.title("Distributin of Store Competition Distance")


# #### **The distribution is right skewed, so we'll replace missing values with the median.**

# In[38]:


# replace missing values in CompetitionDistance with median for the store dataset

store.competitiondistance.fillna(store.competitiondistance.median(), inplace=True)


# **Pairplot for Store Dataset**

# **Checking stores with their assortment type**

# In[39]:


#checking stores with their assortment type 
sns.set_style("whitegrid")
fig, ax = plt.subplots()
fig.set_size_inches(11, 7)
store_type=sns.countplot(x='storetype',hue='assortment', data=store,palette="inferno")

for p in store_type.patches:
    store_type.annotate(f'\n{p.get_height()}', (p.get_x()+0.15, p.get_height()),ha='center', va='top', color='white', size=10)


# ***We can see that there is not such significant differences in these 3 years in terms of sales.*** 

# In[40]:


#plotting year vs sales
sns.catplot(x='year',y='sales',data=store_train, height=4, aspect=4 );


# ### **Merging Two Datasets**

# In[41]:


train = pd.merge(store_train, store, on='store', how='left')
train.head()

test = pd.merge(store_test, store, on='store', how='left')


# In[42]:


train.info()


# ## **EDA On Merged Dataset**

# #### **Heatmap Of Merged Dataset**

# In[43]:


numeric_columns = train.select_dtypes(include=[np.number])

corr_matrix = numeric_columns.corr().abs()

plt.subplots(figsize=(20, 12))
sns.heatmap(corr_matrix, annot=True)
plt.show()


# In[44]:


train["avg_customer_sales"] = train.sales/train.customers


# In[45]:


f, ax = plt.subplots(2, 3, figsize = (20,10))

store.groupby("storetype")["store"].count().plot(kind = "bar", ax = ax[0, 0], title = "Total StoreTypes in the Dataset")
train.groupby("storetype")["sales"].sum().plot(kind = "bar", ax = ax[0,1], title = "Total Sales of the StoreTypes")
train.groupby("storetype")["customers"].sum().plot(kind = "bar", ax = ax[0,2], title = "Total nr Customers of the StoreTypes")
train.groupby("storetype")["sales"].mean().plot(kind = "bar", ax = ax[1,0], title = "Average Sales of StoreTypes")
train.groupby("storetype")["avg_customer_sales"].mean().plot(kind = "bar", ax = ax[1,1], title = "Average Spending per Customer")
train.groupby("storetype")["customers"].mean().plot(kind = "bar", ax = ax[1,2], title = "Average Customers per StoreType")

plt.subplots_adjust(hspace = 0.3)
plt.show()


# **As we can see from the graphs, the StoreType A has the most stores, sales and customers. However the StoreType D has the best averages spendings per customers. StoreType B, with only 17 stores has the most average customers.**

# #### **Lets go ahead with the promotions**

# In[46]:


sns.catplot(data=train, x="month", y="sales", 
            col='promo', # per store type in cols
            hue='promo2', 
            row="year", 
            kind="point", 
            height=5, aspect=1.5)

plt.subplots_adjust(top=0.9)
plt.suptitle("Sales by Month, Promo, and Year", fontsize=16)

plt.show()
# So, of course, if the stores are having promotion the sells are higher.
# Overall the store promotions sellings are also higher than the seasionality promotions (Promo2). However I can't see no yearly trend. 


# **As We can see that when the promo is running Sales are high**

# In[47]:


sns.catplot(data=train, x="dayofweek", y="sales", hue="promo", kind="point", height=6, aspect=1.5)

plt.title("Sales by Day of Week and Promo", fontsize=16)
plt.show()


# In[48]:


print("""So, no promotion in the weekend. However, the sales are very high, if the stores have promotion. 
The Sales are going crazy on Sunday. No wonder.""")
print("There are", train[(train.open == 1) & (train.dayofweek == '7')].store.unique().shape[0], "stores opend on sundays")    


# **Let's see the trends on a yearly basis.**

# In[49]:


sns.catplot(data=train, x="month", y="sales", col="year", hue="storetype", kind="point", height=6, aspect=1.5)

plt.subplots_adjust(top=0.9)
plt.suptitle("Sales by Month, Year, and Store Type", fontsize=16)

plt.show()
# Yes, we can see a seasonalities, but not trends. The sales stays constantly yearly. 


# In[50]:


train['competitiondistance'].isna().sum()


# **What about the Competition Distance. What kind of inpact does this have on the sales.**

# In[51]:


train = train.dropna(subset=['competitiondistance'])
train['competitiondistance'] = pd.to_numeric(train['competitiondistance'], errors='coerce')
# The obsverations are continous numbers, so we need to convert them into a categories. Lets a create a new variable.
train["competitiondistance_cat"] = pd.cut(train["competitiondistance"], bins=5)

print(train['competitiondistance_cat'].head())


# In[52]:


f, ax = plt.subplots(1,2, figsize = (15,5))

train.groupby(by = "competitiondistance_cat").sales.mean().plot(kind = "bar", title = "Average Total Sales by Competition Distance", ax = ax[0])
train.groupby(by = "competitiondistance_cat").customers.mean().plot(kind = "bar", title = "Average Total Customers by Competition Distance", ax = ax[1])

# It is pretty clear. If the competions is very far away, the stores are performing better (sales and customers)


# In[53]:


train.drop(['avg_customer_sales','competitiondistance_cat'],axis=1,inplace=True)


# **Box plot shows that we have a very high outliers in sales**

# In[54]:


#checking outliers in sales
sns.boxplot(train['sales'])


# **Removing Outliers Of Sales Column**

# In[55]:


#removing outliers
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# In[56]:


# defining new variable after removing outliers
train= remove_outlier(train, 'sales')


# # **Conclusion of the analysis:**
#  
# **Sales are highly correlated to number of Customers.**
#  
# **The most selling and crowded store type is A.**
#  
# **StoreType B has the lowest Average Sales per Customer. So i think customers visit this type only for small things.**
#  
# **StoreTybe D had the highest buyer cart.**
#  
# **Promo runs only in weekdays.**
#  
# **For all stores, Promotion leads to increase in Sales and Customers both.**
#  
# **More stores are opened during School holidays than State holidays.**
#  
# **The stores which are opened during School Holiday have more sales than normal days.**
#  
# **Sales are increased during Chirstmas week, this might be due to the fact that people buy more beauty products during a Christmas celebration.**
#  
# **Promo2 doesnt seems to be correlated to any significant change in the sales amount.**
#  
# **Absence of values in features CompetitionOpenSinceYear/Month doesn’t indicate the absence of competition as CompetitionDistance values are not null where the other two values are null.**

# ### **Drop Subsets Of Data Where Might Cause Bias**

# In[57]:


# where stores are closed, they won't generate sales, so we will remove that part of the dataset
train = train[train.open != 0]


# In[58]:


# Open isn't a variable anymore, so we'll drop it too
train = train.drop('open', axis=1)


# In[59]:


# Check if there's any opened store with zero sales
train['store'] = pd.to_numeric(train['store'], errors='coerce')
train[train.sales == 0]['store'].sum()


# In[60]:


# see the percentage of open stored with zero sales
train[train.sales == 0]['sales'].sum()/train.sales.sum()


# In[61]:


# remove this part of data to avoid bias
train = train[train.sales != 0]


# In[62]:


train_new=train.copy()


# In[63]:


train_new = pd.get_dummies(train_new,columns=['storetype','assortment'])

train_new['promo2'] = train_new['promo2'].replace({'true': 1, 'false': 0})

train_new['storetype_a'] = train_new['storetype_a'].replace({True: 1, False: 0})
train_new['storetype_b'] = train_new['storetype_b'].replace({True: 1, False: 0})
train_new['storetype_c'] = train_new['storetype_c'].replace({True: 1, False: 0})
train_new['storetype_d'] = train_new['storetype_d'].replace({True: 1, False: 0})

train_new['assortment_a'] = train_new['assortment_a'].replace({True: 1, False: 0})
train_new['assortment_b'] = train_new['assortment_b'].replace({True: 1, False: 0})
train_new['assortment_c'] = train_new['assortment_c'].replace({True: 1, False: 0})


# In[64]:


train_new.head()


# **From plot it can be sen that most of the sales have been on 1st and last day of week**

# In[65]:


train_new['dayofweek'] = pd.to_numeric(train_new['dayofweek'], errors='coerce')

colors = sns.color_palette("husl", 7)  # 7 unique colors for 7 days

plt.figure(figsize=(15, 8))
sns.barplot(x='dayofweek', y='sales', data=train_new, palette=colors)

plt.xlabel('Day of the Week')
plt.ylabel('Sales')
plt.title('Sales in Terms of Days of the Week')
plt.show()


# #### **Setting Features and Target Variables**

# In[66]:


X = train_new.drop(['sales', 'store', 'year'] , axis = 1)
y= train_new.sales


# In[67]:


X.shape


# In[68]:


X.head()


# In[69]:


X.head()


# In[70]:


y.head()


# Splitting Dataset Into Training Set and Test Set

# In[141]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)


# In[142]:


columns=X_train.columns


# ## **Implementing Supervised Machine Learning algorithms.**

# ## **1.  Linear Regression (OLS)**

# In[143]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


import joblib

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

joblib.dump(regressor, 'linear_regression_model.pkl')


# In[145]:


regressor.intercept_


# In[146]:


regressor.coef_


# In[147]:


y_pred_train = regressor.predict(X_train)


# In[148]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)


# In[149]:


def rmse(x, y):
    return sqrt(mean_squared_error(x, y))

# definte MAPE function
def mape(x, y): 
    return np.mean(np.abs((x - y) / x)) * 100

print("Training RMSE", ":", rmse(y_train, y_pred_train),
      "Testing RMSE", ":", rmse(y_test, y_pred))
print("Training MAPE", ":", mape(y_train, y_pred_train),
      "Testing MAPE", ":", mape(y_test, y_pred))


# In[150]:


mean_squared_error(y_test, y_pred)


# In[151]:


# Test performance
math.sqrt(mean_squared_error(y_test, y_pred))


# In[152]:


train_score_1=regressor.score(X_train,y_train)
train_score_1


# In[153]:


test_score_1=regressor.score(X_test,y_test)
test_score_1


# In[83]:


#storing 100 observations for analysis
simple_lr_pred = y_pred[:100]
simple_lr_real = y_test[:100]
dataset_lr = pd.DataFrame({'Real':simple_lr_real,'PredictedLR':simple_lr_pred}) #storing these values into dataframe


# In[84]:


#storing absolute diffrences between actual sales price and predicted
dataset_lr['diff']=(dataset_lr['Real']-dataset_lr['PredictedLR']).abs()


# In[85]:


#visualising our predictions
sns.lmplot(x='Real', y='PredictedLR', data=dataset_lr, line_kws={'color': 'black'});


# ## **Inferences On Linear Regression Coefficients**

# In[86]:


X = X.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
y = pd.to_numeric(y, errors='coerce')


# In[87]:


print(X.dtypes)
print(y.dtypes)


# In[88]:


X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
model.summary()


# ## **2. LARS Lasso Regression**

# In[89]:


las = LassoLars(alpha=0.3, fit_intercept=False)
lasreg = las.fit(X_train, y_train)

joblib.dump(lasreg, 'lasso_lars_model.pkl')


# In[90]:


def rmse(x, y):
    return sqrt(mean_squared_error(x, y))

# definte MAPE function
def mape(x, y): 
    return np.mean(np.abs((x - y) / x)) * 100

train_score_2=lasreg.score(X_train, y_train)
test_score_2=lasreg.score(X_test, y_test)

print("Regresion Model Score" , ":" , train_score_2 , "," ,
      "Out of Sample Test Score" ,":" , test_score_2)

y_predicted = lasreg.predict(X_train)
y_test_predicted = lasreg.predict(X_test)

print("Training RMSE", ":", rmse(y_train, y_predicted),
      "Testing RMSE", ":", rmse(y_test, y_test_predicted))
print("Training MAPE", ":", mape(y_train, y_predicted),
      "Testing MAPE", ":", mape(y_test, y_test_predicted))


# ## **3. Decision Tree Regression**

# In[91]:


tree = DecisionTreeRegressor()
treereg = tree.fit(X_train, y_train)

joblib.dump(treereg, 'decision_tree_model.pkl')


# In[92]:


train_score_3=treereg.score(X_train, y_train)
test_score_3=treereg.score(X_test, y_test)

print("Regresion Model Score" , ":" , train_score_3 , "," ,
      "Test Score" ,":" , test_score_3)

y_predicted = treereg.predict(X_train)
y_test_predicted = treereg.predict(X_test)
print("Training RMSE", ":", rmse(y_train, y_predicted),
      "Testing RMSE", ":", rmse(y_test, y_test_predicted))
print("Training MAPE", ":", mape(y_train, y_predicted),
      "Testing MAPE", ":", mape(y_test, y_test_predicted))


# ### **Decision Tree With Hyper Parameter Tuning**

# In[93]:


# #another script that takes toooo long, to find the right parameters for tree
# tree = DecisionTreeRegressor()

# params = {
#          'min_samples_split':[2,3,5,7],
#          'min_samples_leaf':[6,8,10],
#          }

# grid = RandomizedSearchCV(estimator=rfr,param_distributions=params,verbose=True,cv=10)
# #choosing 10 K-Folds makes sure i went through all of the data and didn't miss any pattern.

# grid.fit(X_train, y_train)
# grid.best_params_


#  **I trained Model with hyper parameters..to not run everytime i record the result**
# 
# **Here are our best parameters for Decision Tree**
# 
# **{ min_samples_split=5,min_samples_leaf=8 }**

# In[94]:


tree = DecisionTreeRegressor(min_samples_leaf=8,min_samples_split=5)
treereg = tree.fit(X_train, y_train)

joblib.dump(treereg, 'tuned_decision_tree_model.pkl')


# In[95]:


train_score_4=treereg.score(X_train, y_train)
test_score_4=treereg.score(X_test, y_test)

print("Regresion Model Score" , ":" , train_score_4 , "," ,
      "Test Score" ,":" , test_score_4)

y_predicted = treereg.predict(X_train)
y_test_predicted = treereg.predict(X_test)
print("Training RMSE", ":", rmse(y_train, y_predicted),
      "Testing RMSE", ":", rmse(y_test, y_test_predicted))
print("Training MAPE", ":", mape(y_train, y_predicted),
      "Testing MAPE", ":", mape(y_test, y_test_predicted))


# In[96]:


#storing 100 observations for analysis
dc_pred = y_test_predicted[:100]
dc_real = y_test[:100]
dataset_dc = pd.DataFrame({'Real':dc_real,'PredictedDC':dc_pred}) #storing these values into dataframe


# In[97]:


#storing absolute diffrences between actual sales price and predicted
dataset_dc['diff']=(dataset_dc['Real']-dataset_dc['PredictedDC']).abs()


# In[98]:


#visualising our predictions
sns.lmplot(x='Real', y='PredictedDC', data=dataset_dc, line_kws={'color': 'black'});


# # **4. Support Vector Regression**

# In[99]:


#%%time
#from sklearn.svm import SVR
#svr=SVR()
#svr_reg=svr.fit(X_train,y_train)

#joblib.dump(svr_reg, 'svr_model.pkl')

#print("Regresion Model Score" , ":" , svr_reg.score(X_train, y_train) , "," ,
#      "Out of Sample Test Score" ,":" , svr_reg.score(X_test, y_test))

#y_predicted = svr_reg.predict(X_train)
#y_test_predicted = svr_reg.predict(X_test)

#print("Training RMSE", ":", rmse(y_train, y_predicted),
#      "Testing RMSE", ":", rmse(y_test, y_test_predicted))
#print("Training MAPE", ":", mape(y_train, y_predicted),
#      "Testing MAPE", ":", mape(y_test, y_test_predicted))


# ## **5. K-Nearest Neighbors Regression**

# In[100]:


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 30)
knnreg = knn.fit(X_train, y_train)

joblib.dump(knnreg, 'knn_model.pkl')


# In[ ]:


#train_score_knn=knnreg.score(X_train, y_train)
#test_score_knn=knnreg.score(X_test, y_test)

print("Regresion Model Score" , ":" , knnreg.score(X_train, y_train) , "," ,
      "Out of Sample Test Score" ,":" , knnreg.score(X_test, y_test))

y_predicted = knnreg.predict(X_train)
y_test_predicted = knnreg.predict(X_test)

print("Training RMSE", ":", rmse(y_train, y_predicted),
      "Testing RMSE", ":", rmse(y_test, y_test_predicted))
print("Training MAPE", ":", mape(y_train, y_predicted),
      "Testing MAPE", ":", mape(y_test, y_test_predicted))


# ### **6. Random Forest With Hyper Parameter Tuning**

# In[ ]:


# #another script that takes toooo long, to find the right parameters for RFR
# rfr=RandomForestRegressor(n_jobs=-1)

# params = {
#          'n_estimators':[40,50,60,70,80,90],
#          'min_samples_split':[2,3,6,8],
#          'min_samples_leaf':[1,2,3,4],
#          'max_depth':[None,5,15,30]
#          }

# #the dimensionality is high, the number of combinations we have to search is enormous, using RandomizedSearchCV is a better option then GridSearchCV
# grid = RandomizedSearchCV(estimator=rfr,param_distributions=params,verbose=True,cv=10)

# #choosing 10 K-Folds makes sure i went through all of the data and didn't miss any pattern.
# grid.fit(X_train, y_train)
# grid.best_params_


#  **I trained Model with hyper parameters..to not run everytime i record the result**
# 
# **Here are our best parameters for Random Forest**
# 
# **{ n_estimators=80,min_samples_split=2,min_samples_leaf=1,max_depth=None }**

# In[102]:


#%%time
rdf = RandomForestRegressor(n_estimators=80,min_samples_split=2, min_samples_leaf=1,max_depth=None,n_jobs=-1)
rdfreg = rdf.fit(X_train, y_train)

joblib.dump(rdfreg, 'random_forest_model.pkl')


# In[103]:


train_score_5=rdfreg.score(X_train, y_train)
test_score_5=rdfreg.score(X_test, y_test)

print("Regresion Model Score" , ":" , train_score_5 , "," ,
      "Test Score" ,":" , test_score_5)   

y_predicted_2 = rdfreg.predict(X_train)
y_test_predicted_2 = rdfreg.predict(X_test)

print("Training RMSE", ":", rmse(y_train, y_predicted_2),
      "Testing RMSE", ":", rmse(y_test, y_test_predicted_2))
print("Training MAPE", ":", mape(y_train, y_predicted_2),
      "Testing MAPE", ":", mape(y_test, y_test_predicted_2))


# In[115]:


y_test_predicted_2


# In[116]:


#storing 100 observations for analysis
rf_prd = y_test_predicted_2[:100]
rf_real = y_test[:100]
dataset_rf = pd.DataFrame({'Real':rf_real,'PredictedRF':rf_prd})


# In[117]:


#storing absolute diffrences between actual sales price and predicted
dataset_rf['diff']=(dataset_rf['Real']-dataset_rf['PredictedRF']).abs()


# In[118]:


# taking 4 sample
dataset_rf.sample(4)


# In[119]:


#Statistical description of our predictions and actual values 
dataset_rf.describe()


# In[120]:


plt.style.use('ggplot')
# Plotting the histogram for Actual, Predicted, and Difference of sales
dataset_rf.plot.hist(subplots=True, layout=(3, 1), legend=False)
plt.show()


# **As we can see that Actual, Prediction values are approximately closed to each other and there is no such significant variation in our plots.**

# In[122]:


print(dataset_rf.columns)


# In[123]:


#visualising our predictions
sns.lmplot(x='Real', y='PredictedRF', data=dataset_rf, line_kws={'color': 'red'}, height=6, aspect=1)


# ## **Feature Importance On Random Forest Regressor**
# 
# As we can see that Random Forest has the highest test score

# In[126]:


def plot_feature_importance(importance,names,model_type):

  #Create arrays from feature importance and feature names
  feature_importance = np.array(importance)
  feature_names = np.array(names)

  #Create a DataFrame using a Dictionary
  data={'feature_names':feature_names,'feature_importance':feature_importance}
  fi_df = pd.DataFrame(data)

  #Sort the DataFrame in order decreasing feature importance
  fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

  #Define size of bar plot
  plt.figure(figsize=(10,8))
  #Plot Searborn bar chart
  colors = sns.color_palette("viridis", len(fi_df))
  sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'], palette=colors)
  #Add chart labels
  plt.title(model_type + ' FEATURE IMPORTANCE')
  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')


# In[127]:


plot_feature_importance(rdfreg.feature_importances_,columns[:],'RANDOM FOREST')


# **Customers, CompetitionDistance, StoreType_d, Promo these four are most important features in our sales prediction.**

# In[160]:


train_score_knn=0.6769432117695016 #forgot during training, thus assigning manually
test_score_knn=0.651167641097711


# In[170]:


score_df = pd.DataFrame({'Train_Score':[train_score_1,train_score_2,train_score_3,train_score_4,train_score_knn,train_score_5],'Test_Score':[test_score_1,test_score_2,test_score_3,test_score_4,test_score_knn,test_score_5]},index=['Linear Regression','Lasso Regression','Decision Tree',"Decision Tree(hyperparameters)",'KNN','Random Forest Regression'])


# In[171]:


score_df.reset_index(inplace=True)
score_df.rename(columns={'index': 'Model'}, inplace=True)

# Melt the dataframe for seaborn compatibility
score_df_melted = score_df.melt(id_vars=['Model'], value_vars=['Train_Score', 'Test_Score'], 
                                var_name='Score_Type', value_name='Score')

# Plotting
plt.figure(figsize=(10, 6))
sns.set_palette("Set1")  # Change color palette
sns.barplot(x='Model', y='Score', hue='Score_Type', data=score_df_melted)

# Customize the plot
plt.title('Train vs Test Scores for Different Models', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Score Type')

# Show the plot
plt.tight_layout()
plt.show()


# # **Conclusion**

# In[163]:


score_df


# **Random Forest regressor achieved lowest MAPE as 5.65% showing that it is a highly accurate model. MAE is the average magnitude of error produced by your model, the MAPE is how far the model’s predictions are off from their corresponding outputs on average**![ndIXERr.jpg](data:image/jpeg;base64,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)

# In[174]:


score_df_mape = pd.DataFrame({
    'Train_RMSE': [1148.5155712691585, 1148.5663125136766, 0.0, 475.9603158051788, 1403.6600848568366, 195.67066329604697],
    'Test_RMSE': [1145.7200607239986, 1145.7661292396926, 738.0357000513005, 654.1943902989904, 1455.903948654709, 515.863430260663],
    'Train_MAPE': [14.128762809238363, 14.13336720582164, 0.0, 5.03618980365554, 18.084037703407, 2.1060676268713863],
    'Test_MAPE': [14.080610509749569, 14.08507704, 7.733286662647336, 6.942528087053865, 18.72932102522484, 5.6187714979872565]
}, index=['Linear Regression', 'Lasso Regression', 'Decision Tree', 'Decision Tree (Hyperparameter)', 'KNN', 'Random Forest'])

# Resetting index and renaming for compatibility with seaborn
score_df_mape.reset_index(inplace=True)
score_df_mape.rename(columns={'index': 'Model'}, inplace=True)

# Melting the dataframe for seaborn compatibility
score_df_melted_mape = score_df_mape.melt(id_vars=['Model'], value_vars=['Train_MAPE', 'Test_MAPE'], 
                                var_name='Metric', value_name='Value')

# Plotting the comparison of all metrics (RMSE and MAPE)
plt.figure(figsize=(12, 6))
sns.set_palette("Set1")  # Change color palette
sns.barplot(x='Model', y='Value', hue='Metric', data=score_df_melted_mape)

# Customize the plot
plt.title('Model Comparison MAPE for Train and Test', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Score Value', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Metric')

# Show the plot
plt.tight_layout()
plt.show()


# In[173]:


score_df_melted_mape = score_df_mape.melt(id_vars=['Model'], value_vars=['Train_RMSE', 'Test_RMSE'], 
                                var_name='Metric', value_name='Value')

# Plotting the comparison of all metrics (RMSE and MAPE)
plt.figure(figsize=(12, 6))
sns.set_palette("Set1")  # Change color palette
sns.barplot(x='Model', y='Value', hue='Metric', data=score_df_melted_mape)

# Customize the plot
plt.title('Model Comparison: RMSE for Train and Test', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Score Value', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Metric')

# Show the plot
plt.tight_layout()
plt.show()

