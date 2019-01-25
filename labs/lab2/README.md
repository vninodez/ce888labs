# ce888labs

Exercise 1 - Plotting

The code for this exercise can be found in the folder Exercise 1 - Plotting, in the file vehicles.py.

First, we create an histogram on the Current Fleet and compute the main descriptive statistics:

![logo](./HistogramCurrentFleet.png?raw=true)

Mean: 20.144578
Median: 19.000000
Var: 40.983113
std: 6.401805
MAD: 4.000000

Second, we create an histogram on the New Fleet and compute the main descriptive statistics:

![logo](./HistogramNewFleet.png?raw=true)

Mean: 30.481013
Median: 32.000000
Var: 36.831918
std: 6.068931
MAD: 4.000000

A scatterplot is also computed:

![logo](./scaterplot.png?raw=true)

Exercise 2 - Bootstrap

The code for this exercise can be found in the folder Exercise 2 - Bootstrap, in the file bootstrap.py.

First, the function boostrap() is created. 
Then this is used to compare the mean values of the MPG for each Fleet at a 95% CI.

The results of this analysis are: 

Current Fleet Mean: 20.161739
Current Fleet Lower: 19.321185
Current Fleet Upper: 20.960040

New Fleet Mean: 30.481823
New Fleet Lower: 29.176899
New Fleet Upper: 31.887025
