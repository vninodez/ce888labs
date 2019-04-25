import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C://Users//vnino//Desktop//CE888 - Assignment 2//Python code//mushrooms')

names=['label', 'cap_shape', 'cap_surface', 
       'cap_color', 'bruises', 'odor', 
       'gill_attachment', 'gill_spacing', 'gill_size',
       'gill_color', 'stalk_shape', 'stalk_root', 
       'stalk_surface_above', 'stalk_surface_below', 
       'stalk_color_above', 'stalk_color_below', 'veil_type',
       'veil_color', 'ring_number', 'ring_type',
       'spore_print_color', 'population','habitat']
df = pd.read_csv("agaricus-lepiota.csv", names=names)

#Checking shape and head of the data
df.shape
df.head()

# Descriptive stats
df.describe()

#Distribution of categories of the features
for var in df.columns:
    print(df[var].value_counts()/8124)
    
# Plot of classes distribution
plot1=df['label'].value_counts().plot(kind='barh', color='0.75', title='Number of instances by edibility label', figsize=(7,3));

for i, v in enumerate(df['label'].value_counts()):
   plot1.text(v + 2, i-0.15 , str(v), fontsize='large')
plot1