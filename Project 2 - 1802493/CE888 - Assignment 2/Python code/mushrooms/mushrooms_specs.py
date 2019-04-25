# Import packages
import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.chdir('C://Users//vnino//Desktop//CE888 - Assignment 2//Python code//mushrooms//data')
import random

# Fetch the dataset
names=['label', 'cap_shape', 'cap_surface', 
       'cap_color', 'bruises', 'odor', 
       'gill_attachment', 'gill_spacing', 'gill_size',
       'gill_color', 'stalk_shape', 'stalk_root', 
       'stalk_surface_above', 'stalk_surface_below', 
       'stalk_color_above', 'stalk_color_below', 'veil_type',
       'veil_color', 'ring_number', 'ring_type',
       'spore_print_color', 'population','habitat']
df = pd.read_csv("agaricus-lepiota.csv", names=names)

# Filter missing values
df=df[df.stalk_root!='?']

df=df.drop(columns=['veil_type', 'veil_color'])

data_df=pd.get_dummies(df, prefix=['label', 'cap_shape', 'cap_surface', 
       'cap_color', 'bruises', 'odor', 
       'gill_attachment', 'gill_spacing', 'gill_size',
       'gill_color', 'stalk_shape', 'stalk_root', 
       'stalk_surface_above', 'stalk_surface_below', 
       'stalk_color_above', 'stalk_color_below',
       'ring_number', 'ring_type',
       'spore_print_color', 'population','habitat'])

data_df=data_df.drop(columns=['label_e'])
data_df=data_df.rename(index=str, columns={'label_p': 'label'})

#Defining labels and features
data_df.reset_index(inplace=True, drop=True)
target=data_df['label']
data=np.asarray(data_df.iloc[:,21:])
n_samples= data.shape[0]
n_clusters=2

# Get the split between training/test set and validation set
random.seed(0)
test_indices = random.sample(range(0, 5644), 1120)
validation_indices = [x for x in range(0, 5644) if x not in test_indices]

# Auto-encoder architecture
input_size = data.shape[1]
hidden_1_size = 500
hidden_2_size = 500
hidden_3_size = 2000
embedding_size = n_clusters
dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensions
              hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None, # Encoder layer activations # None is linear activation
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None] # Decoder layer activations # None is linear activation
names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
         'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names

activations_soft = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.softmax, # Encoder layer activations # Uses softmax in last enocder layer
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None]


#data_df['label'].value_counts()