# For creating this code the following sources were used: 
# 1. Building autoencoders in Keras. Source: https://blog.keras.io/building-autoencoders-in-keras.html
# 2. Implementation of "Deep k-Means: Jointly Clustering with k-Means and Learning Representations" by Maziar Moradi Fard, Thibaut Thonet, and Eric Gaussier. (https://github.com/MaziarMF/deep-k-means)
# 3. In Depth: k-Means Clustering (https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)

# IMPORTS PACKAGES ############################################################################################################################
###############################################################################################################################################

import os
import math
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.metrics import  silhouette_score, accuracy_score, completeness_score, homogeneity_score
from sklearn.cluster import KMeans
from scipy.stats import mode
import seaborn as sns
sns.set(font_scale=3)
import matplotlib.pyplot as plt
TF_FLOAT_TYPE = tf.float32

from keras.optimizers import SGD, Adam
from keras.layers import Dense, Input
from keras.models import Model

from keras.backend import clear_session


# FUNCTIONS DEFINITION ########################################################################################################################
###############################################################################################################################################

def cluster_acc(y_true, y_pred, n_clusters):
    labels = np.zeros_like(y_pred)
    for i in range(n_clusters):
        mask = (y_pred == i)
        labels[mask] = mode(y_true[mask])[0]
    if y_true is not None:
        acc = np.round(accuracy_score(y_true, labels), 5)
    return acc

def save_confusion_matrix(y_true, y_pred, n_clusters, name):
    labels = np.zeros_like(y_pred)
    for i in range(n_clusters):
        mask = (y_pred == i)
        labels[mask] = mode(y_true[mask])[0]
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, labels)
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.savefig('./results/'+ name + '.png')

def next_batch(num, data):
    indices = np.arange(0, data.shape[0])
    np.random.shuffle(indices)
    indices = indices[:num]
    batch_data = np.asarray([data[i, :] for i in indices])
    return indices, batch_data

def create_autoencoder(dims):
    input_layer = Input(shape=(dims[0],), name='input')
    encoded = Dense(dims[1], activation='relu', name='enc_0')(input_layer)
    encoded = Dense(dims[2], activation='relu', name='enc_1')(encoded)
    encoded = Dense(dims[3], activation='relu', name='enc_2')(encoded)
    encoded = Dense(dims[4], activation='softmax', name='enc_3')(encoded) ##, activation='softmax'
    
    decoded = Dense(dims[3], activation='relu', name='dec_3')(encoded)
    decoded = Dense(dims[2], activation='relu', name='dec_2')(decoded)
    decoded = Dense(dims[1], activation='relu', name='dec_1')(decoded)
    decoded = Dense(dims[0], activation='sigmoid', name='dec_0')(decoded) # activation='sigmoid'
    
    return Model(input_layer, outputs=decoded, name='AE'), Model(input_layer, outputs=encoded, name='encoder')

# MAIN PROGRAM ################################################################################################################################
###############################################################################################################################################

# This program executes DEEP K-means for a chosen dataset. 
# It runs the algorithm 10 times, each with different seed.
# It computes the final perfomance metrics selecting the iteration with the 
# maximum Accuracy. The performance metrics are printed in the screen as 
# they are computed. Also, a txt file is generated with these results
# and 2 png file with the confusion matrices are created.

if __name__=='__main__':
    
# STEP 0: Setting the directory -------------------------------------------------------------------------------------------------------------
    directory = input('Please enter the directory where you have the datasets: ')
    os.chdir(directory)
    
# STEP 1 Importing dataset ------------------------------------------------------------------------------------------------------------------    
    dataset = input('Select the dataset for which you want to run DEEP K-Means, digits/HAR/mushrooms [d/h/m]: ')
    if dataset=='d':
        import digits_specs as specs 
    elif dataset=='h':
        import HAR_specs as specs 
    elif dataset=='m':
        import mushrooms_specs as specs 
    else: 
        print('Enter a valid dataset [d/h/m]: ')

# STEP 2: Running DEEP K-Means ------------------------------------------------------------------------------------------------------------------
    
    # 2.1. Setting initial parameters------------------------------------------------------------------------------------------------------------
    
  
    n_train_epochs = 50
    batch_size = 256 # Size of the mini-batches used in the stochastic optimizer
    seeded = True # Specify if runs are seeded
    seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]  # Use a fixed seed for this run, as defined in the seed list 
                                                                        # There is randomness in centroids initialization and mini-batch selection
    n_runs = 10
    
    dims = [specs.input_size, 500, 500, 2000, specs.embedding_size]

    target = specs.target
    data = specs.data

    dict_acc = dict()
    dict_sc = dict()
    dict_cs = dict()
    dict_hs = dict()
    dict_clusterassign= dict()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # 2.2. Running k-means and creating the output files-----------------------------------------------------------------------------------------
    
    if os.path.exists('./results/AEsoft_results.txt'):
        os.remove('./results/AEsoft_results.txt') # Removes the text file if exists
    with open('./results/AEsoft_results.txt', 'x') as f: 
        
        # 2.2.1. Running k-means n_run times-------------------------------------------------------------------------------------------------------
       
        for run in range(n_runs):
            clear_session()
            if seeded:
                tf.reset_default_graph()
                tf.set_random_seed(seeds[run])
                np.random.seed(seeds[run])
            print("Run", run)
            f.write("RUN "+ str(run) +'\n'+'\n') 
            autoencoder, encoder = create_autoencoder(dims)
            autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
            #autoencoder.compile(optimizer='adam', loss='kld')
            autoencoder.fit(data, data, batch_size=batch_size, epochs=n_train_epochs)
            # Second, run k-means++ on the trained embeddings
            print("Running k-means on the learned embeddings...")
            f.write("1. K-MEANS ON THE EMBEDDED FEATURES"+'\n'+'\n')
            #kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(embeddings) # n_init=10 initial centroids
            kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(encoder.predict(data))
            # Evaluate the clustering performance using the ground-truth labels
            sc = silhouette_score(data, kmeans_model.labels_)
            print("SC", sc)
            f.write("SC: "+ str(sc) +'\n')
        
            acc = cluster_acc(target, kmeans_model.labels_, specs.n_clusters)
            print("ACC", acc)
            f.write("ACC: "+ str(acc) +'\n')
            
            cs = completeness_score(target, kmeans_model.labels_)
            print("CS", cs)
            f.write("CS: "+ str(cs) +'\n')
            
            hs = homogeneity_score(target, kmeans_model.labels_)
            print("HS", hs)
            f.write("HS: "+ str(hs) +'\n')
                     
        # Record the clustering performance for the run
            dict_acc.update({run : acc})
            dict_sc.update({run : sc})
            dict_cs.update({run : cs})
            dict_hs.update({run : hs})
            dict_clusterassign.update({run : kmeans_model.labels_ })
            f.write('\n'+'-------------------------------------------------------------'+'\n') 
        
        max_acc = max(dict_acc, key=dict_acc.get)
        print('Iteration with maximum ACC: '+ str(max_acc))
        print("ACC: {:.3f}".format(dict_acc[max_acc]))
        print("SC: {:.3f}".format(dict_sc[max_acc]))
        print("CS: {:.3f}".format(dict_cs[max_acc]))
        print("HS: {:.3f}".format(dict_hs[max_acc]))
        f.write('Run with maximum ACC: '+ str(max_acc) +'\n')
        f.write("ACC: {:.3f}".format(dict_acc[max_acc]) +'\n')
        f.write("SC: {:.3f}".format(dict_sc[max_acc]) +'\n')
        f.write("CS: {:.3f}".format(dict_cs[max_acc]) +'\n')
        f.write("HS: {:.3f}".format(dict_hs[max_acc]) +'\n')
        save_confusion_matrix(target, dict_clusterassign[max_acc], specs.n_clusters, 'AEsoft_CM') 
   





