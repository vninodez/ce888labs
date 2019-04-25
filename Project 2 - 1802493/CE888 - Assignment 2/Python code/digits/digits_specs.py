# Import packages
import os
import tensorflow as tf
from sklearn import datasets
os.chdir('C://Users//vnino//Desktop//CE888 - Assignment 2//Python code//digits')
import random

# Fetch the dataset
dataset = datasets.load_digits()
data = dataset.data
target = dataset.target
n_samples = data.shape[0] # Number of obs in the dataset
n_clusters = 10 # Number of clusters to obtain

# Pre-process the dataset
data = data / 16.0 # Normalize the levels of grey between 0 and 1

# Get the split between training/test set and validation set
random.seed(0)
test_indices = random.sample(range(0, 1797), 360)
validation_indices = [x for x in range(0, 1797) if x not in test_indices]

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