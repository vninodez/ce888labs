#Source: https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_EDA.ipynb

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('C://Users//vnino//Desktop//CE888 - Assignment 2//Python code//HAR')

# Fetch the dataset
train_df= pd.read_csv("train.csv")
test_df= pd.read_csv("test.csv")
data_df=pd.concat([train_df,test_df], ignore_index=True)

#Defining labels and features
target=data_df['Activity']
data=np.asarray(data_df.iloc[:,0:560])
n_samples= data.shape[0]


#Distribution of the mean of Linear Body Acceleration Euclidean Norm (tBodyAccMagmean)
sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(data_df, hue='ActivityName', size=6,aspect=2)
facetgrid.map(sns.distplot,'tBodyAccMagmean', hist=False)\
    .add_legend()
plt.annotate("Stationary Activities", xy=(-0.956,17), xytext=(-0.9, 23), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

plt.annotate("Moving Activities", xy=(0,3), xytext=(0.2, 9), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.show()

#Boxplot of the mean of Linear Body Acceleration Euclidean Norm (tBodyAccMagmean)
sns.boxplot(x='ActivityName', y='tBodyAccMagmean', data = data_df, showfliers=False)
plt.xticks(rotation = 40)
plt.axhline(y=-0.22, xmin=0.1, xmax=0.8, dashes=(5,3), c='m')
plt.xticks(rotation=90)
plt.show()

#Boxplot of the mean of Angle Y-Gravity (tBodyAccMagmean)
sns.boxplot(x='ActivityName', y='angleYgravityMean', data = data_df, showfliers=False)
plt.xticks(rotation = 40)
plt.axhline(y=-0.22, xmin=0.1, xmax=0.8, dashes=(5,3), c='m')
plt.xticks(rotation=90)
plt.show()