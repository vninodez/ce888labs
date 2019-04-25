# Import packages
import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.chdir('C://Users//vnino//Desktop//CE888 - Assignment 2//Python code//HAR//data')
import random

# Fetch the dataset
train_df= pd.read_csv("train.csv")
test_df= pd.read_csv("test.csv")
data_df=pd.concat([train_df,test_df], ignore_index=True)

#Defining labels and features
target=data_df['Activity']
data=np.asarray(data_df.iloc[:,0:560])
n_samples= data.shape[0]
n_clusters=6

# Get the split between training/test set and validation set
random.seed(0)
test_indices = random.sample(range(0, 10299), 2060)
validation_indices = [x for x in range(0, 10299) if x not in test_indices]

# Auto-encoder architecture
input_size = data.shape[1]
hidden_1_size = 500
hidden_2_size = 500
hidden_3_size = 2000
embedding_size = n_clusters
dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensions
              hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None, # Encoder layer activations # None uses linear act.
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None] # Decoder layer activations # None uses linear act.
names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
         'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names

activations_soft = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.softmax, # Encoder layer activations # Uses softmax in last enocder layer
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None]

#data_df['ActivityName'].value_counts()
