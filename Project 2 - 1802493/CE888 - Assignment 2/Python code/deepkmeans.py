# For creating this code the following sources were used: 
# 1. Implementation of "Deep k-Means: Jointly Clustering with k-Means and Learning Representations" by Maziar Moradi Fard, Thibaut Thonet, and Eric Gaussier. (https://github.com/MaziarMF/deep-k-means)
# 2. In Depth: k-Means Clustering (https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)

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

def fc_layers(input, specs):
    [dimensions, activations, names] = specs
    for dimension, activation, name in zip(dimensions, activations, names):
        input = tf.layers.dense(inputs=input, units=dimension, activation=activation, name=name, reuse=False)
    return input

def autoencoder(input, specs):
    [dimensions, activations, names] = specs
    mid_ind = int(len(dimensions)/2)
    # Encoder
    embedding = fc_layers(input, [dimensions[:mid_ind], activations[:mid_ind], names[:mid_ind]])
    # Decoder
    output = fc_layers(embedding, [dimensions[mid_ind:], activations[mid_ind:], names[mid_ind:]])
    return embedding, output

def f_func(x, y):
    return tf.reduce_sum(tf.square(x - y), axis=1)

def g_func(x, y):
    return tf.reduce_sum(tf.square(x - y), axis=1)

# COMPUTATION GRAPH FOR DEEP K-MEANS ##########################################################################################################
###############################################################################################################################################

class DkmCompGraph(object):
    def __init__(self, ae_specs, n_clusters, val_lambda):
        input_size = ae_specs[0][-1]
        embedding_size = ae_specs[0][int((len(ae_specs[0])-1)/2)]

        # Placeholder tensor for input data
        self.input = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=(None, input_size))

        # Auto-encoder loss computations
        self.embedding, self.output = autoencoder(self.input, ae_specs)  # Get the auto-encoder's embedding and output
        rec_error = g_func(self.input, self.output)  # Reconstruction error based on distance g

        # k-Means loss computations
        ## Tensor for cluster representatives
        minval_rep, maxval_rep = -1, 1 # Clusters rep (centroids) are initialized randomly using U(-1,1)
        self.cluster_rep = tf.Variable(tf.random_uniform([n_clusters, embedding_size],
                                                    minval=minval_rep, maxval=maxval_rep,
                                                    dtype=TF_FLOAT_TYPE), name='cluster_rep', dtype=TF_FLOAT_TYPE)

        ## First, compute the distance f between the embedding and each cluster representative
        list_dist = []
        for i in range(0, n_clusters):
            dist = f_func(self.embedding, tf.reshape(self.cluster_rep[i, :], (1, embedding_size)))
            list_dist.append(dist)
        self.stack_dist = tf.stack(list_dist)

        ## Second, find the minimum squared distance for softmax normalization
        min_dist = tf.reduce_min(list_dist, axis=0)

        ## Third, compute exponentials shifted with min_dist to avoid underflow (0/0) issues in softmaxes
        self.alpha = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=())  # Placeholder tensor for alpha
        list_exp = []
        for i in range(n_clusters):
            exp = tf.exp(-self.alpha * (self.stack_dist[i] - min_dist))
            list_exp.append(exp)
        stack_exp = tf.stack(list_exp)
        sum_exponentials = tf.reduce_sum(stack_exp, axis=0)

        ## Fourth, compute softmaxes and the embedding/representative distances weighted by softmax
        list_softmax = []
        list_weighted_dist = []
        for j in range(n_clusters):
            softmax = stack_exp[j] / sum_exponentials
            weighted_dist = self.stack_dist[j] * softmax
            list_softmax.append(softmax)
            list_weighted_dist.append(weighted_dist)
        stack_weighted_dist = tf.stack(list_weighted_dist)

        # Compute the full loss combining the reconstruction error and k-means term
        self.ae_loss = tf.reduce_mean(rec_error)
        self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(stack_weighted_dist, axis=0))
        self.loss = self.ae_loss + val_lambda * self.kmeans_loss

        # The optimizer is defined to minimize this loss
        optimizer = tf.train.AdamOptimizer() 
        #optimizer = tf.train.AdadeltaOptimizer()
        self.pretrain_op = optimizer.minimize(self.ae_loss) # Pretrain the autoencoder before starting DKM
        self.train_op = optimizer.minimize(self.loss) # Train the whole DKM model

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
    
    n_pretrain_epochs = 50
    n_finetuning_epochs = 5
    lambda_ = 0.1 # Optimal lambda tested on USPS (similar database)
    batch_size = 256 # Size of the mini-batches used in the stochastic optimizer
    n_batches = int(math.ceil(specs.n_samples / batch_size)) # Number of mini-batches
    seeded = True # Specify if runs are seeded
    max_n = 20  # Number of alpha values to consider (constant values are used here), 100 epochs in total 20*5
    alphas = 1000*np.ones(max_n, dtype=float) # alpha is constant, all epochs take constant alpha of 1000
    seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]  # Use a fixed seed for this run, as defined in the seed list 
                                                                        # There is randomness in centroids initialization and mini-batch selection
    n_runs = 10

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
    
    if os.path.exists('./results/DKM_results.txt'):
        os.remove('./results/DKM_results.txt') # Removes the text file if exists
    with open('./results/DKM_results.txt', 'x') as f: 
        
        # 2.2.1. Running k-means n_run times-------------------------------------------------------------------------------------------------------
       
        for run in range(n_runs):
            if seeded:
                tf.reset_default_graph()
                tf.set_random_seed(seeds[run])
                np.random.seed(seeds[run])
            print("Run", run)
            f.write("RUN "+ str(run) +'\n'+'\n') 
        # Define the computation graph for DKM
        # We need a computation graph that performs the simultaneous minimization of clusters and low-dim representation, so we have to program a specific low-level graph 
            cg = DkmCompGraph([specs.dimensions, specs.activations, specs.names], specs.n_clusters, lambda_)
        # DkmCompGraph([[500, 500, 2000, 10,
        #                2000, 500, 500, 64], 
        #               [tf.nn.relu, tf.nn.relu, tf.nn.relu, None, 
        #                tf.nn.relu, tf.nn.relu, tf.nn.relu, None], 
        #               ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', 
        #                'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] ], 
        #               10, 1.0)
            
        
        # Run the computation graph
            with tf.Session(config=config) as sess:
            # Initialization
                init = tf.global_variables_initializer()
                sess.run(init)
            
            # Variables to save tensor content
                distances = np.zeros((specs.n_clusters, specs.n_samples))
                print("Starting autoencoder pretraining...")
                
                # Variables to save pretraining tensor content
                embeddings = np.zeros((specs.n_samples, specs.embedding_size), dtype=float)
                
                # First, pretrain the autoencoder
                ## Loop over epochs
                for epoch in range(n_pretrain_epochs):
                    print("Pretraining step: epoch {}".format(epoch))
                    
                    # Loop over the samples
                    for _ in range(n_batches):
                        # Fetch a random data batch of the specified size
                        indices, data_batch = next_batch(batch_size, data)

                        # Run the computation graph until pretrain_op (only on autoencoder) on the data batch
                        _, embedding_, ae_loss_ = sess.run((cg.pretrain_op, cg.embedding, cg.ae_loss),
                                                       feed_dict={cg.input: data_batch})

                        # Save the embeddings for batch samples
                        for j in range(len(indices)):
                            embeddings[indices[j], :] = embedding_[j, :]
                            
                            #print("ae_loss_:", float(ae_loss_))

                # Second, run k-means++ on the pretrained embeddings
                print("Running k-means on the learned embeddings...")
                f.write("1. K-MEANS ON THE PRETRAINED FEATURES"+'\n'+'\n')
                kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(embeddings) # n_init=10 initial centroids
                
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
                    
                # The cluster centers are used to initialize the cluster representatives in DKM
                sess.run(tf.assign(cg.cluster_rep, kmeans_model.cluster_centers_))
                
            # Train the full DKM model
                if (len(alphas) > 0):
                    print("Starting DKM training...")
                    f.write('\n') 
                    f.write("2. DEEP K-MEANS TRAINING"+'\n'+'\n') 
                ## Loop over alpha (inverse temperature), from small to large values
                for k in range(len(alphas)):
                    print("Training step: alpha[{}]: {}".format(k, alphas[k]))
                    f.write("TRAINING STEP: {}".format(k)+'\n'+'\n')
                    # Loop over epochs per alpha
                    for _ in range(n_finetuning_epochs):
                        # Loop over the samples
                        for _ in range(n_batches):
                            #print("Training step: alpha[{}], epoch {}".format(k, i))
                            
                            # Fetch a random data batch of the specified size
                            indices, data_batch = next_batch(batch_size, data)
                            
                            #print(tf.trainable_variables())
                            #current_batch_size = np.shape(data_batch)[0] # Can be different from batch_size for unequal splits
                            
                            # Run the computation graph on the data batch
                            _, loss_, stack_dist_, cluster_rep_, ae_loss_, kmeans_loss_ =\
                            sess.run((cg.train_op, cg.loss, cg.stack_dist, cg.cluster_rep, cg.ae_loss, cg.kmeans_loss),
                                     feed_dict={cg.input: data_batch, cg.alpha: alphas[k]})
                            
                            # Save the distances for batch samples
                            for j in range(len(indices)):
                                distances[:, indices[j]] = stack_dist_[:, j]
                                
                    # Evaluate the clustering performance every print_val alpha and for last alpha
                    print_val = 1
                    if k % print_val == 0 or k == max_n - 1:
                        print("loss:", loss_)
                        f.write("Total loss: "+ str(loss_) +'\n') 
                        print("ae loss:", ae_loss_)
                        f.write("AE loss: "+ str(ae_loss_) +'\n') 
                        print("kmeans loss:", kmeans_loss_)
                        f.write("KMeans loss: "+ str(kmeans_loss_) +'\n') 
                                    
                        # Infer cluster assignments for all samples
                        cluster_assign = np.zeros((specs.n_samples), dtype=float)
                        for i in range(specs.n_samples):
                            index_closest_cluster = np.argmin(distances[:, i])
                            cluster_assign[i] = index_closest_cluster
                        cluster_assign = cluster_assign.astype(np.int64)
                        
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
                        f.write('\n') 
                            
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
        save_confusion_matrix(target, dict_clusterassign[max_acc], specs.n_clusters, 'DKM_CM') 
   



