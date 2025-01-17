import pandas as pd
import numpy as np

def power(sample1, sample2, reps, perm_iterations, alpha):
    tobs=np.mean(sample1)-np.mean(sample2)
    m=0
    for r in range(reps):
        n=0
        for i in range(perm_iterations):
            join_sample=np.append(sample1, sample2)
            resample1=np.random.choice(join_sample, size=sample1.shape[0], replace=True)
            resample2=np.random.choice(join_sample, size=sample2.shape[0], replace=True)
            if (np.mean(resample1)-np.mean(resample2)) > tobs:
                n+=1
        p_value=n/perm_iterations
        print(p_value)
        if p_value<alpha:
            m+=1
    return(m/reps)

sample2=np.arange(0,1000,1)
sample1=np.arange(0,1000,1)

power(sample1,sample2,100,100,0.05)         
    
np.mean(sample1)-np.mean(sample2)  

join_sample=np.append(sample1, sample2)
resample1=np.random.choice(join_sample, size=sample1.shape[0], replace=True)
resample2=np.random.choice(join_sample, size=sample2.shape[0], replace=True)

np.mean(resample1)-np.mean(resample2)

    
