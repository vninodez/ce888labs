import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np



def boostrap(sample, sample_size, iterations,ciparam):
    sample_matrix=np.zeros((iterations,sample_size))
    data_mean_mx=np.zeros(iterations)
    for i in range(iterations):
        sample_matrix[i, :]=np.random.choice(sample, size=sample_size, replace=True)
        data_mean_mx[i]=np.mean(sample_matrix[i, :])
    lower=np.percentile(data_mean_mx,ciparam/2)
    upper=np.percentile(data_mean_mx,100-ciparam/2)
    data_mean=np.mean(data_mean_mx)
    print(data_mean_mx)
    return data_mean, lower, upper

sample=np.arange(10,1000,2)
sample_size=100
iterations=10
boostrap(sample, sample_size, iterations)


if __name__ == "__main__":
	df = pd.read_csv('./salaries.csv')

	data = df.values.T[1]
	boots = []
	for i in range(100, 100000, 1000):
		boot = boostrap(data, data.shape[0], i)
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
	


	