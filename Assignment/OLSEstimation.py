#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cleands import least_squares_regressor
import statsmodels.formula.api as smf


np.random.seed(1)
sample = [5,10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
rho_list = [  0.80,0.9,-0.1, -0.2]
df = pd.DataFrame()
ax = plt.subplot(111)
for rho in rho_list:
    bias_list = []
    for n in sample:
        x = np.random.normal(size=(n, 1))
        y = np.zeros(shape=(n,))
        for t in range(1,n):
            y[t] = rho*y[t-1]+x[t]
        ones = np.ones(shape=(n-1,1))
        x = y[:-1].reshape(-1,1)
        y = y[1:]
        x = np.hstack((ones, x))
        model = least_squares_regressor(x, y)
        bias = rho - model.params[1]
        bias_list.append(bias)
        df = df.append({"Bias": bias, "Size_Inv": 1 / n, "Rho": rho}, ignore_index=True)
    ax.plot(sample, bias_list, label= 'rho = {}'.format(rho))

#Plot Bias vs Sample
plt.title('Bias Vs Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('Bias')
ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
plt.show()

#Data     
pd.Series(y).plot();

#find optimum values of parameters
ols_model = smf.ols(formula='Bias ~ Rho + Size_Inv + Size_Inv:Rho', data=df).fit()
print(ols_model.summary().tables[1])

