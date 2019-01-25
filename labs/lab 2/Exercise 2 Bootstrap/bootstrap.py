import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np

## PART 1: Creating the boostrap function

def boostrap(sample, sample_size, iterations,ciparam):
    sample_matrix=np.zeros((iterations,sample_size))
    data_mean_mx=np.zeros(iterations)
    for i in range(iterations):
        sample_matrix[i, :]=np.random.choice(sample, size=sample_size, replace=True)
        data_mean_mx[i]=np.mean(sample_matrix[i, :])
    lower=np.percentile(data_mean_mx,ciparam/2)
    upper=np.percentile(data_mean_mx,100-ciparam/2)
    data_mean=np.mean(data_mean_mx)
    return data_mean, lower, upper

## PART 2 Applying the boostrap function on the vehicles data

vehicles_df = pd.read_csv('./vehicles.csv')
vehicles_df.columns=['Currentfleet','Newfleet']
vehicles_df2=vehicles_df[vehicles_df.Newfleet.notnull()]
current_fleet_data = vehicles_df.values.T[0]
new_fleet_data = vehicles_df2.values.T[1]

cf=boostrap(current_fleet_data, current_fleet_data.shape[0],1000,5)

print(("Current Fleet Mean: %f")%(cf[0]))
print(("Current Fleet Lower: %f")%(cf[1]))
print(("Current Fleet Upper: %f")%(cf[2]))

nf=boostrap(new_fleet_data, new_fleet_data.shape[0],1000,5)

print(("New Fleet Mean: %f")%(nf[0]))
print(("New Fleet Lower: %f")%(nf[1]))
print(("New Fleet Upper: %f")%(nf[2]))

## The New fleet has a higher MPG mean, the 95% CI for the mean of the new fleet 
# is between 29.1 MPG and 31.7 MPG, while for the Current Fleet it is between 19.4 and 20.9.

## Plots on salaries using boostrap iterations

if __name__ == "__main__":
	df = pd.read_csv('./salaries.csv')

	data = df.values.T[1]
	boots = []
	for i in range(100, 100000, 1000):
		boot = boostrap(data, data.shape[0], i, 5)
		boots.append([i, boot[0], "mean"])
		boots.append([i, boot[1], "lower"])
		boots.append([i, boot[2], "upper"])

	df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
	sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

	sns_plot.axes[0, 0].set_ylim(0,)
	sns_plot.axes[0, 0].set_xlim(0, 100000)

	sns_plot.savefig("bootstrap_confidence.png", bbox_inches='tight')
	sns_plot.savefig("bootstrap_confidence.pdf", bbox_inches='tight')


	#print ("Mean: %f")%(np.mean(data))
	#print ("Var: %f")%(np.var(data))
	



	