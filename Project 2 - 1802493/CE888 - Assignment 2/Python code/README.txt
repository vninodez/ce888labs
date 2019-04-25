INSTRUCTIONS FOR RUNNING THE CLUSTERING
-----------------------------------------------------------------------

1. SETTING PATH FOR INPUT DATA: 

The input data sets are stored in the input folders: 
- digits
- HAR
- mushrooms

Before running the algorithms, you will need to edit the specs files 
(HAR_specs.py, mushrooms_specs.py) inside these folders to change the
path of the text files. The text files are stored in the same folders 
as the specs files.  

2. RUNNING CLUTERING:

Each clustering technique is programmed in a separate .py file: 
- kmeans.py
- autoencoder.py
- autoencodersoftmax.py
- deepkmeans.py

You can run each of these programs on your command windows or using 
a python IDE. For each run the program will ask to insert the path 
where the specs files are stored and to specify the dataset you want 
to process.

3. OUTPUT: 

The run has 2 files as output written in the folder "results" that can 
be found inside of each the input folders. The files are: a .txt file 
containing the results (performance scores) of the run and a .png file 
containing the confusion matrix for the clusters obtained.

The folders already contain the results of running the codes. Re-running
it, will over write the files.

ADDITIONAL FILES
-----------------------------------------------------------------------

Inside the input folders, you can also found the code used in assignment 1
to make the previous EDA of the three data sets.
