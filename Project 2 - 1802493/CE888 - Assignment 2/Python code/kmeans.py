# For creating this code the following sources were used: 
# 1. Implementation of "Deep k-Means: Jointly Clustering with k-Means and Learning Representations" by Maziar Moradi Fard, Thibaut Thonet, and Eric Gaussier. (https://github.com/MaziarMF/deep-k-means)
# 2. In Depth: k-Means Clustering (https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)

# IMPORTS PACKAGES ############################################################################################################################
###############################################################################################################################################

import os
import numpy as np
import sklearn
from sklearn.metrics import silhouette_score, accuracy_score, completeness_score, homogeneity_score
from sklearn.cluster import KMeans
from scipy.stats import mode
import seaborn as sns
sns.set(font_scale=3)
import matplotlib.pyplot as plt

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

# MAIN PROGRAM ################################################################################################################################
###############################################################################################################################################

# This program executes the traditional K-means for a chosen dataset. 
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
    dataset = input('Select the dataset for which you want to run K-Means, digits/HAR/mushrooms [d/h/m]: ')
    if dataset=='d':
        import digits_specs as specs 
    elif dataset=='h':
        import HAR_specs as specs 
    elif dataset=='m':
        import mushrooms_specs as specs 
    else: 
        print('Enter a valid dataset [d/h/m]: ')

# STEP 2: Running K-Means ------------------------------------------------------------------------------------------------------------------
    
    # 2.1. Setting initial parameters------------------------------------------------------------------------------------------------------------

    seeded = True # Specify if runs are seeded
    seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575] #Specific seeds used
    n_runs = 10 # How many times should we run the algorithm
    
    dict_acc = dict()
    dict_sc = dict()
    dict_cs = dict()
    dict_hs = dict()
    dict_clusterassign= dict()
    
    target = specs.target
    data = specs.data

    # 2.2. Running k-means and creating the output files-----------------------------------------------------------------------------------------
    
    if os.path.exists('./results/KM_results.txt'):
        os.remove('./results/KM_results.txt') # Removes the text file if exists
    with open('./results/KM_results.txt', 'x') as f: 
        
        # 2.2.1. Running k-means n_run times-------------------------------------------------------------------------------------------------------
        
        for run in range(n_runs):
            np.random.seed(seeds[run])
            # Shuffle the dataset
            print("Run", run)
            f.write("RUN "+ str(run) +'\n'+'\n') 
            
            # Run k-means(++) on the original data

            kmeans_model = KMeans(n_clusters=specs.n_clusters, init="k-means++").fit(data)
                
            # Evaluate the clustering
            
            cluster_assign = np.asarray(kmeans_model.labels_)
                
            # Evaluate the clustering
            
            sc = silhouette_score(data, cluster_assign)
            print("SC", sc)
            f.write("SC: "+ str(sc) +'\n')
                
            acc = cluster_acc(target, cluster_assign, specs.n_clusters)
            print("ACC", acc)
            f.write("ACC: "+ str(acc) +'\n')
                
            cs = completeness_score(target, cluster_assign)
            print("CS", cs)
            f.write("CS: "+ str(cs) +'\n')
                
            hs = homogeneity_score(target, cluster_assign)
            print("HS", hs)
            f.write("HS: "+ str(hs) +'\n')
                
            # Save performance scores and predictions
            
            dict_acc.update({run : acc})
            dict_sc.update({run : sc})
            dict_cs.update({run : cs})
            dict_hs.update({run : hs})
            dict_clusterassign.update({run : cluster_assign})
            
            f.write('\n'+'-------------------------------------------------------------'+'\n') 
                
        # 2.2.2. Getting the iteration with the best Accuracy Score---------------------------------------------------------------------------------
        
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
        save_confusion_matrix(target, dict_clusterassign[max_acc], specs.n_clusters, 'KM_CM') 
    
 