#!/usr/bin/env python
# coding: utf-8

# ## Appliances Energy Consumption Prediction

# In[12]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[13]:


df = pd.read_csv(r"C:\Users\akash\OneDrive\Documents\dataset\kaggle dataset\Appliances Energy Consumption.zip",parse_dates=['date'])
df


# In[14]:


df.shape


# In[5]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


df.hist(bins=50, figsize=(20,15))
plt.savefig("atribute_histogram_points")
plt.show()


# In[9]:


df.head()


# In[15]:


df.columns = [x.lower() for x in df.columns]


# In[16]:


df = df.set_index('date')


# In[12]:


df.head()


# In[13]:


df.corr()


# In[16]:


import seaborn as sns


# In[15]:


# sns.pairplot(df)


# In[10]:


# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
#

# In[17]:


corrmap = df.corr()
top_corr_features = corrmap.index
plt.figure(figsize=(20,20))
#making of heatmap
df_heatmap = sns.heatmap(df[top_corr_features].corr(),annot=True, cmap="RdYlGn")


# In[17]:


sorted_appliances = df.sort_values('appliances',ascending=False)
sorted_appliances.head()


# In[19]:


len(sorted_appliances.head(len(sorted_appliances)//1000))


# In[20]:


sorted_appliances.appliances[19]


# In[18]:


sorted_appliances = df.sort_values('appliances',ascending=False)
print("The number of the 0,1% top values of appliances' load is ",
     len(sorted_appliances.head(len(sorted_appliances)//1000)),"and they have power load higher than",
     sorted_appliances.appliances[19],"Wh,")

#boxplot appliances
sns.set(style = "whitegrid")
ax = sns.boxplot(sorted_appliances.appliances)


# In[19]:


df = df.dropna()
df = df.drop(df[(df.appliances>790)|(df.appliances<0)].index)


# In[20]:


sorted_appliances = df.sort_values('appliances',ascending=False)
sorted_appliances.head()


# In[21]:


import warnings
warnings.filterwarnings("ignore")


# In[22]:


df['hour'] = df.index.hour
#df['week'] = df.index.week
df['weekday'] = df.index.weekday
df['month'] = df.index.month


# In[23]:


import numpy as np

df['log_appliances']= np.log(df.appliances)


# In[24]:


df['house_temp'] = (df.t1+df.t2+df.t3+df.t4+df.t5+df.t6+df.t7+df.t8+df.t9)/8
df['house_hum'] = (df.rh_1+df.rh_2+df.rh_3+df.rh_4+df.rh_5+df.rh_6+df.rh_7+df.rh_8+df.rh_9)/8


# In[25]:


df['house_temp'].head()


# In[26]:


df['house_hum'].head()


# In[27]:


# remove additive assumptions

df['house"lights'] = df.hour + df.lights
df['t1rh1'] = df.t1 * df.rh_1
df['t2rh2'] = df.t2 * df.rh_2
df['t3rh3'] = df.t3 * df.rh_3
df['t4rh4'] = df.t4 * df.rh_4
df['t5rh5'] = df.t5 * df.rh_5
df['t6rh6'] = df.t6 * df.rh_6
df['t7rh7'] = df.t7 * df.rh_7
df['t8rh8'] = df.t8 * df.rh_8
df['t9rh9'] = df.t9 * df.rh_9


# In[28]:


def code_mean(data, cat_feature, real_feature):
    return dict(data.groupby(cat_feature)[real_feature].mean())


# In[29]:


df['weekday_avg'] = list(map(
    code_mean(df[:],'weekday','appliances').get,df.weekday))
df['hour_avg'] = list(map(
    code_mean(df[:],'hour','appliances').get,df.hour))


# In[30]:


df['weekday_avg'].head()


# In[31]:


df['hour_avg'].head()


# In[32]:


df_hour = df.resample('1H').mean()
df_30min = df.resample('30min').mean()


# In[33]:


df_hour.head()


# In[34]:


df_30min.head()


# In[35]:


# setting the assumptions as to lower or higher
# tryouts

df_hour['low_consum'] = (df_hour.appliances+25<(df_hour.hour_avg))*1
df_hour['high_consum'] = (df_hour.appliances+25>(df_hour.hour_avg))*1

df_30min['low_consum'] = (df_30min.appliances+25<(df_30min.hour_avg))*1
df_30min['high_consum'] = (df_30min.appliances+35<(df_30min.hour_avg))*1


# In[36]:


def daily(x,df=df):
    return df.groupby('weekday')[x].mean()
def hourly(x,df=df):
    return df.groupby('hour')[x].mean()

def monthly_daily(x,df=df):
    by_day = df.pivot_table(index='weekday',
                           columns=['month'],
                           values=x,
                           aggfunc='mean')

    return round(by_day, ndigits=2)


# In[37]:


# ploting the hourly consumtion
hourly('appliances').plot(figsize=(10,9))
plt.xlabel('Hour')
plt.ylabel('Appliances consumptions in Wh')
ticks = list(range(0,24,1))
plt.title('Mean Energy Consumption per Hour of a Day')

plt.xticks(ticks);


# In[41]:


#Weekly Consumption

daily('appliances').plot(kind = 'bar',color=['pink', 'red', 'green', 'blue', 'cyan', 'yellow', 'orange'],figsize=(10,7))
ticks = list(range(0, 7, 1))
labels = "Mon Tues Weds Thurs Fri Sat Sun". split()
plt. xlabel ('Day')
plt.ylabel('Appliances consumption in Wh' )
plt. title( 'Mean Energy Consumption per Day of Week' )
plt.xticks(ticks, labels)
plt.show()


# In[42]:


#MOnthly Consumption
sns.set(rc={'figure.figsize': (10,8)},)
ax=sns.heatmap(monthly_daily('appliances').T, cmap="PiYG",
            xticklabels="Mon Tues Weds Thurs Fri Sat Sun".split(),
            yticklabels="January February March April May".split(),
            annot=True ,fmt='g',
            cbar_kws={'label':'Consmption in wH'}).set_title("Mean appliances consumption(wh) per weekday/month")
plt.show()


# In[43]:


f, axes = plt. subplots(1, 2, figsize=(10,5))

sns.distplot(df_hour.appliances, hist=True, color = 'red', hist_kws={'edgecolor': 'black'},ax=axes [0])
axes[0].set_title("Appliance's Consumption")
axes [0].set_xlabel('Appliances wH')

sns.distplot(df_hour.log_appliances, hist=True, color = 'green', hist_kws={'edgecolor': 'black'},ax=axes [1])
axes[1].set_title("Log Appliance's Consumption")
axes[1].set_xlabel('Appliances Log(wH) ')


# In[44]:


col = ['log_appliances','lights','t1','rh_1','t2','rh_2','t3','rh_3','t4','rh_4','t5','rh_5','t6','rh_6',
      't7','rh_7','t8','rh_8','t9','rh_9','t_out','press_mm_hg','rh_out','windspeed','visibility','tdewpoint','hour']
corr = df[col].corr()
plt.figure(figsize=(18,18))
sns.set(font_scale=1)
sns.heatmap(corr, cbar=True, annot=True, cmap='RdYlGn', fmt= '.2f',xticklabels=col, yticklabels=col)
plt.show()


# In[45]:


col = ['t6','t2','rh_2','lights','hour','t_out','windspeed','tdewpoint']
sns.set(style= 'ticks',color_codes=True)
sns.pairplot(df[col])
plt.show()


# **Traing the Model**

# In[38]:


for cat_feature in ['weekday','hour']:
    df_hour = pd.concat([df_hour,pd.get_dummies(df_hour[cat_feature])],axis=1)
    df_30min = pd.concat([df_30min,pd.get_dummies(df_30min[cat_feature])],axis=1)
    df = pd.concat([df,pd.get_dummies(df[cat_feature])],axis=1)



# In[39]:


lin_model = ['low_consum','high_consum','hour','t6','rh_6','lights','windspeed','t6rh6']


# In[40]:


df_hour.lights = df_hour.lights.astype(float)
df_hour.log_appliances = df_hour.log_appliances.astype(float)
df_hour.hour = df_hour.hour.astype(float)
df_hour.low_consum = df_hour.low_consum.astype(float)
df_hour.high_consum = df_hour.high_consum.astype(float)
df_hour.t6rh6 = df_hour.t6rh6.astype(float)


# In[41]:


test_size=.2
test_index = int (len(df_hour.dropna())*(1-test_size))

X1_train, X1_test = df_hour[lin_model].iloc[:test_index,], df_hour[lin_model].iloc[test_index:,]
y1_train = df_hour.log_appliances.iloc[:test_index,]
y_test = df_hour. log_appliances. iloc[test_index:, ]


# In[42]:


from sklearn.preprocessing import StandardScaler

scaler =StandardScaler()
scaler.fit(X1_train)
X1_train = scaler.transform(X1_train)
X1_test = scaler.transform(X1_test)


# In[55]:


from sklearn import linear_model

lin_model = linear_model.LinearRegression()
lin_model.fit(X1_train,y1_train)


# **Model Evaluation**

# In[56]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import cross_val_predict,cross_val_score


# In[57]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    r_score = 100 * r2_score(test_labels, predictions)
    accuracy = 100 - mape
    print(model, '\n')
    print('Average Error             : {:0.4f} degrees'.format(np.mean(errors)))
    print('Variance score R^2        : {:0.2f}% '.format(r_score))
    print('Accuracy                  : {:0.2f}%\n'.format(accuracy))


# In[58]:


evaluate(lin_model,X1_test,y_test)


# Model Evaluation
#
# - Average Error             : 0.3605 degrees
# - Variance score R^2        : 11.80%
# - Accuracy                  : 91.59%

# In[ ]:





# In[6]:


from sklearn.linear_model import Ridge


# In[49]:


rid = Ridge()
rid.fit(X1_train,y1_train)


# In[50]:


y_rid_pred = rid.predict(X1_test)


# In[51]:


from sklearn.metrics import r2_score,mean_absolute_error

r2_score(y_test,y_rid_pred)


# In[52]:


mean_absolute_error(y_test,y_rid_pred)

features = ['low_consum', 'high_consum', 'hour', 't6', 'rh_6', 'lights', 'windspeed', 't6rh6']

# In[ ]:

import pickle

# Save the model
with open('lin_model.pkl', 'wb') as f:
    pickle.dump(lin_model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# During training
with open('columns.pkl', 'wb') as f:
    pickle.dump(features, f)

