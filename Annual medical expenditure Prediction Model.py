#!/usr/bin/env python
# coding: utf-8

# ## Annual medical expenditure Prediction Model 

# #### Downloading the Data

# In[1]:


get_ipython().system('pip install pandas-profiling --quiet')


# In[2]:


medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'


# In[3]:


from urllib.request import urlretrieve


# In[4]:


urlretrieve(medical_charges_url, 'medical.csv')


# In[5]:


import pandas as pd


# In[6]:


medical_df = pd.read_csv('medical.csv')


# In[7]:


medical_df


# Our objective is to find a way to estimate the value in the "charges" column using the values in the other columns. If we can do so for the historical data, then we should able to estimate charges for new customers too, simply by asking for information like their age, sex, BMI, no. of children, smoking habits and region.

# In[8]:


medical_df.info()


# As observed "age", "children", "bmi" (body mass index) and "charges" are numbers, whereas "sex", "smoker" and "region" are strings (possibly categories). None of the columns contain any missing values

# In[9]:


medical_df.describe()


# ### Exploratory Analysis and Visualization

# In[10]:


get_ipython().system('pip install plotly matplotlib seaborn --quiet')


# In[11]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# ### Age

# In[13]:


medical_df.age.describe()


# Age is a numeric column. The minimum age in the dataset is 18 and the maximum age is 64. Thus, we can visualize the distribution of age using a histogram with 47 bins (one for each year) and a box plot

# In[14]:


fig = px.histogram(medical_df, 
                   x='age', 
                   marginal='box', 
                    nbins=47,
                   title='Distribution of Age')
fig.update_layout(bargap=0.1)
fig.show()


# The distribution of ages in the dataset is almost uniform, with 20-30 customers at every age, except for the ages 18 and 19, which seem to have over twice as many customers as other ages. The uniform distribution might arise from the fact that there isn't a big variation in the number of people of any given age (between 18 & 64) 

# ### Body Mass Index

# In[15]:


medical_df.bmi.describe()


# BMI is a numeric (float 64) column. The minimum bmi in the dataset is 15.96 and the maximum age is 53.13. Thus, we can visualize the distribution of age using a histogram and a box plot.
# 
# 

# In[16]:


fig = px.histogram(medical_df, 
                   x='bmi', 
                   marginal='box', 
                   color_discrete_sequence=['red'], 
                   title='Distribution of BMI (Body Mass Index)')
fig.update_layout(bargap=0.1)
fig.show()


# The measurements of body mass index seem to form a Gaussian distribution centered around the value 30, with a few outliers towards the right.

# In[17]:


fig = px.histogram(medical_df, 
                 x='bmi', 
                 y='sex', 
                color='sex', 
                 opacity=1,
                   marginal='box',
                 hover_data=['smoker'],
                 title='Sex vs BMI')

fig.show()


# In[18]:


bmi_male = medical_df.loc[medical_df['sex'] == 'male', 'bmi'].mean()
bmi_female = medical_df.loc[medical_df['sex'] == 'female', 'bmi'].mean()
print('Average BMI for male ',bmi_male)
print('Average BMI for female ',bmi_female)


# In[19]:


fig = px.histogram(medical_df, 
                 x='age', 
                 y='bmi', 
                color='sex', 
                 opacity=0.8,
                   marginal='box',
                 hover_data=['sex'],
                 title='Age vs BMI')
fig.update_layout(bargap=0.1)
fig.show()


# Hence, observation says that average bmi in males is slightly greater than that in females, with largest in the age group of 18-19. This may be because the amount of body fat changes with age and the amount of body fat differs between males and females.

# ### Charges

# In[20]:


medical_df.charges.describe()


# In[21]:


fig = px.histogram(medical_df, 
                   x='charges', 
                    marginal='box',
                   color='smoker', 
                   color_discrete_sequence=['green', 'grey'], 
                   title='Annual Medical Charges')
fig.update_layout(bargap=0.1)
fig.show()


# In[22]:


median_smokers = medical_df.loc[medical_df['smoker'] == 'yes', 'charges'].median()
median_nonsmokers = medical_df.loc[medical_df['smoker'] == 'no', 'charges'].median()
print('median for smokers ', median_smokers)
print('median for non-smokers ', median_nonsmokers)


# We can make the following observations from the above graph:
# 
# - For most customers, the annual medical charges are under dollar 10,000. Only a small fraction of customer have higher medical expenses, possibly due to accidents, major illnesses and genetic diseases. The distribution follows a "power law"
# - There is a significant difference in medical expenses between smokers and non-smokers. While the median for non-smokers is dollar 7300, the median for smokers is close to $35,000.

# In[23]:


medical_df.region.describe()


# In[24]:


southwest_charges = medical_df.loc[medical_df['region'] == 'southwest', 'charges'].sum()
southeast_charges = medical_df.loc[medical_df['region'] == 'southeast', 'charges'].sum()
northwest_charges = medical_df.loc[medical_df['region'] == 'northwest', 'charges'].sum()
northeast_charges = medical_df.loc[medical_df['region'] == 'northeast', 'charges'].sum()
print("Charges in southwest region ", southwest_charges)
print("Charges in southeast region ", southeast_charges)
print("Charges in northwest region ", northwest_charges)
print("Charges in northeast region ", northeast_charges)


# In[25]:


fig = px.bar(medical_df, 
                 x='charges', 
                 y='region', 
                color='smoker', 
                 opacity=1,
                 hover_data=['sex'],
                 title='charges vs. region')

fig.show()


# The graph and the total amount amount of charges region-wise clearly suggest that the medical charges in the southeast region are the highest

# ### Smoker

# In[26]:


medical_df.smoker.value_counts()


# In[27]:


px.histogram(medical_df, x='smoker', color='sex', title='Smoker')


# In[28]:


percentage_smokers = medical_df.loc[medical_df['smoker'] == 'yes','smoker'].count()/medical_df.smoker.count()*100
percentage_smokers


# It appears that 20% of customers have reported that they smoke. We can also see that smoking appears a more common habit among males. 

# ### Age and Charges

# Let's visualize the relationship between "age" and "charges" using a scatter plot. Each point in the scatter plot represents one customer. We'll also use values in the "smoker" column to color the points.

# In[29]:


fig = px.scatter(medical_df, 
                 x='age', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='Age vs. Charges')
fig.update_traces(marker_size=6)
fig.show()


# We can make the following observations from the above chart:
# 
# - The general trend seems to be that medical charges increase with age, as we might expect. However, there is significant variation at every age, and it's clear that age alone cannot be used to accurately determine medical charges.
# 
# - We can see three "clusters" of points, each of which seems to form a line with an increasing slope:
# 
# 1. The first and the largest cluster consists primary of presumably "healthy non-smokers" who have relatively low medical charges compared to others
# 
# 2. The second cluster contains a mix of smokers and non-smokers. It's possible that these are actually two distinct but overlapping clusters: "non-smokers with medical issues" and "smokers without major medical issues".
# 
# 3. The final cluster consists exclusively of smokers, presumably smokers with major medical issues that are possibly related to or worsened by smoking.

# ### BMI and Charges

# In[30]:


fig = px.scatter(medical_df, 
                 x='bmi', 
                 y='charges', 
                 color='smoker', 
                 opacity=0.8, 
                 hover_data=['sex'], 
                 title='BMI vs. Charges')
fig.update_traces(marker_size=5)
fig.show()


# It appears that for non-smokers, an increase in BMI doesn't seem to be related to an increase in medical charges. However, medical charges seem to be significantly higher for smokers with a BMI greater than 30.

# In[31]:


fig = px.violin(medical_df, y="charges", x="smoker", color="sex", box = True, points="all",color_discrete_sequence=['red', 'blue'],
          hover_data=medical_df.columns)
fig.update_traces(marker_size=4)
fig.show()


# ### Correlation

# As you can tell from the analysis, the values in some columns are more closely related to the values in "charges" compared to other columns. E.g. "age" and "charges" seem to grow together, whereas "bmi" and "charges" don't.
# 
# This relationship is often expressed numerically using a measure called the correlation coefficient, which can be computed using the .corr method of a Pandas series

# In[32]:


medical_df.charges.corr(medical_df.age)


# In[33]:


medical_df.charges.corr(medical_df.bmi)


# To compute the correlation for categorical columns, they must first be converted into numeric columns.

# In[34]:


smoker_values = {'no': 0, 'yes': 1}
smoker_numeric = medical_df.smoker.map(smoker_values)
medical_df.charges.corr(smoker_numeric)


# Pandas dataframes also provide a .corr method to compute the correlation coefficients between all pairs of numeric columns.

# In[35]:


medical_df.corr()


# This correlation matrix can be visualised using a heatmap

# In[36]:


sns.heatmap(medical_df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix');


# ## Linear Regression using a Single Feature
# 
# We now know that the "smoker" and "age" columns have the strongest correlation with "charges". Let's try to find a way of estimating the value of "charges" using the value of "age" for non-smokers. First, let's create a data frame containing just the data for non-smokers.

# In[37]:


non_smoker_df = medical_df[medical_df.smoker == 'no']


# Next, let's visualize the relationship between "age" and "charges"

# In[38]:


plt.title('Age vs. Charges')
sns.scatterplot(data=non_smoker_df, x='age', y='charges', alpha=0.7, s=15);


# Model
# In the above case, the x axis shows "age" and the y axis shows "charges". Thus, we're assume the following relationship between the two:
# 
# charges = w x age + b
# 
# We'll try determine w and b for the line that best fits the data.
# 
# - This technique is called linear regression, and we call the above equation a linear regression model, because it models the relationship between "age" and "charges" as a straight line.
# - The numbers w and b are called the parameters or weights of the model.
# - The values in the "age" column of the dataset are called the inputs to the model and the values in the charges column are called "targets".
# 
# Let define a helper function estimate_charges, to compute 'charges'given and 'w' and 'b'.
# 

# In[39]:


def estimate_charges(age, w, b):
    return w * age + b


# The estimate_charges function is our very first model.
# 
# Let's guess the values for w and b and use them to estimate the value for charges.

# In[40]:


w = 50
b = 100


# In[41]:


ages = non_smoker_df.age
estimated_charges = estimate_charges(ages, w, b)


# We can plot the estimated charges using a line graph.

# In[42]:


plt.plot(ages, estimated_charges, 'r-o');
plt.xlabel('Age');
plt.ylabel('Estimated Charges');


# As expected, the points lie on a straight line. 
# 
# We can overlay this line on the actual data, so see how well our _model_ fits the _data_.

# In[43]:


target = non_smoker_df.charges

plt.plot(ages, estimated_charges, 'r', alpha=0.9);
plt.scatter(ages, target, s=8,alpha=0.8);
plt.xlabel('Age');
plt.ylabel('Charges')
plt.legend(['Estimate', 'Actual']);


# Clearly, the our estimates are quite poor and the line does not "fit" the data. However, we can try different values of $w$ and $b$ to move the line around. Let's define a helper function `try_parameters` which takes `w` and `b` as inputs and creates the above plot.

# In[44]:


def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    
    estimated_charges = estimate_charges(ages, w, b)
    
    plt.plot(ages, estimated_charges, 'r', alpha=0.9);
    plt.scatter(ages, target, s=8,alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual']);


# In[45]:


try_parameters(60, 200)


# In[46]:


try_parameters(400, 5000)


# As we change the values, of $w$ and $b$ manually, trying to move the line visually closer to the points, we are _learning_ the approximate relationship between "age" and "charges". 
# 
# Wouldn't it be nice if a computer could try several different values of `w` and `b` and _learn_ the relationship between "age" and "charges"? To do this, we need to solve a couple of problems:
# 
# 1. We need a way to measure numerically how well the line fits the points.
# 
# 2. Once the "measure of fit" has been computed, we need a way to modify `w` and `b` to improve the the fit.
# 
# If we can solve the above problems, it should be possible for a computer to determine `w` and `b` for the best fit line, starting from a random guess.

# ### Loss/Cost Function
# 
# We can compare our model's predictions with the actual targets using the following method:
# 
# * Calculate the difference between the targets and predictions (the differenced is called the "residual")
# * Square all elements of the difference matrix to remove negative values.
# * Calculate the average of the elements in the resulting matrix.
# * Take the square root of the result
# 
# The result is a single number, known as the root mean squared error (RMSE).
# Let's define a function to compute the RMSE.

# In[47]:


get_ipython().system('pip install numpy --quiet')


# In[48]:


import numpy as np


# In[49]:


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


# Let's compute the RMSE for our model with a sample set of weights

# In[50]:


w = 50
b = 100


# In[51]:


try_parameters(w, b)


# In[52]:


targets = non_smoker_df['charges']
predicted = estimate_charges(non_smoker_df.age, w, b)


# In[53]:


rmse(targets, predicted)


# Here's how we can interpret the above number: *On average, each element in the prediction differs from the actual target by \\$8461*. 
# 
# The result is called the *loss* because it indicates how bad the model is at predicting the target variables. It represents information loss in the model: the lower the loss, the better the model.
# 
# Let's modify the `try_parameters` functions to also display the loss.

# In[54]:


def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    predictions = estimate_charges(ages, w, b)
    
    plt.plot(ages, predictions, 'r', alpha=0.9);
    plt.scatter(ages, target, s=8,alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual']);
    
    loss = rmse(target, predictions)
    print("RMSE Loss: ", loss)


# In[55]:


try_parameters(50, 100)


# ### Linear Regression using Scikit-learn

# In[56]:


get_ipython().system('pip install scikit-learn --quiet')


# In[57]:


from sklearn.linear_model import LinearRegression


# In[58]:


model = LinearRegression()


# Next, we can use the `fit` method of the model to find the best fit line for the inputs and targets.

# In[59]:


help(model.fit)


# Not that the input `X` must be a 2-d array, so we'll need to pass a dataframe, instead of a single column.

# In[60]:


inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges
print('inputs.shape :', inputs.shape)
print('targes.shape :', targets.shape)


# Let's fit the model to the data.

# In[61]:


model.fit(inputs, targets)


# We can now make predictions using the model. Let's try predicting the charges for the ages 23, 37 and 61

# In[62]:


model.predict(np.array([[23], 
                        [37], 
                        [61]]))


# Let compute the predictions for the entire set of inputs

# In[63]:


predictions = model.predict(inputs)
predictions


# Let's compute the RMSE loss to evaluate the model.

# In[64]:


rmse(targets, predictions)


# Seems like our prediction is off by $4000 on average, which is not too bad considering the fact that there are several outliers.

# The parameters of the model are stored in the `coef_` and `intercept_` properties.

# In[65]:


# w
model.coef_


# In[66]:


# b
model.intercept_


# Let's visualize the line created by the above parameters.

# In[67]:


try_parameters(model.coef_, model.intercept_)


# Indeed the line is quite close to the points. It is slightly above the cluster of points, because it's also trying to account for the outliers. 
# 
# 

# In[75]:


# Create inputs and targets
inputs, targets = non_smoker_df[['age']], non_smoker_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)


# In[ ]:




