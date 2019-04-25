from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

digits = datasets.load_digits()

print(digits.data.shape)
print(digits.data)
print(digits.target)

digits_df=pd.DataFrame(digits.data)
digits_df.describe()

diglab_df=pd.DataFrame(digits.target)
diglab_df.describe()

# Count of the classes
pd.DataFrame(target)[0].value_counts()

# Plot of classes distribution
plot1=diglab_df[0].value_counts().plot(kind='barh', color='0.75', title='Number of instances by digit', figsize=(7,3));
values=diglab_df[0].value_counts()/1797
values_df=values.to_frame()
values_df['Count']= pd.Series(["{0:.2f}%".format(val * 100) for val in values_df[0]], index = values_df.index)
for i, v in enumerate(diglab_df[0].value_counts()):
   plot1.text(v + 1, i-0.25 , str(v), fontsize='x-small')
plot1

# Plot of the images of the digits
plt.matshow(digits.images[9], cmap='binary')

