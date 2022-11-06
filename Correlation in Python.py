#!/usr/bin/env python
# coding: utf-8

# In[1]:


#first install packages
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

#read in the data
df = pd.read_csv(r'C:\Users\HP\Downloads\movie data\movies.csv')


# In[2]:


df.head()


# In[3]:


#check for missing data
#loop through the data and see if there is anything missing

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[4]:


df = df.dropna(how='any',axis=0) 


# In[5]:


for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[6]:


# Data Types for our columns

print(df.dtypes)


# In[7]:


#change data type of columns to remove decimal point
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')


# In[54]:


df.head()


# In[58]:


#create correct year column
df['yearcorrect'] = df['released'].str.extract(pat = '([0-9]{4})').astype(int)


# In[10]:





# In[11]:


#rearrange 'gross' in descending order
df.sort_values(by=['gross'], inplace=False, ascending = False)


# In[12]:


#takes away empty rows as displayed above
pd.set_option('display.max_rows', None)


# In[13]:


#eliminate duplicates
df.drop_duplicates()


# In[14]:


#are there any outliers?
df.boxplot(column=['gross'])


# In[18]:


#prediction: high correlation between budget and gross
#scatter plot to compare budget and scatter plot

plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Film Budgets')
plt.show()


# In[34]:


#plot includes the regression line usinf seaborn
sns.regplot(x='budget', y='gross', data=df)


# In[38]:


#next determine actual correlation figures
df.corr(method='pearson')


# In[ ]:


#high positive correlation between budget and gross


# In[41]:


#heatmap of correlation matrix

correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot = True)
plt.title("Correlation matrix for Numeric Features")
plt.xlabel("Movie features")
plt.ylabel("Movie features")
plt.show()


# In[42]:


df.head()


# In[43]:


#here, numbers have been assigned to the strings

df_numerized = df


for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name]= df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[44]:


correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[45]:


df_numerized.corr()


# In[50]:


correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs


# In[52]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[53]:


high_corr = sorted_pairs[(sorted_pairs)>0.5]
high_corr


# In[ ]:


#budget and votes have the highest correlation with gross earnings


# In[ ]:




