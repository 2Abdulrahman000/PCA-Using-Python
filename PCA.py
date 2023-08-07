import numpy as np
from numpy import random
from numpy.core.fromnumeric import size
from numpy.random.mtrand import poisson
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
#from sklearn import preprocessing
#from sklearn.decompostion import PCA

#1-Create Dataset with Virtual digital array of numbers for each gene
genes = ['gene'+str(i1) for i1 in range(1,101)]
wt=['wt'+str(i2) for i2 in range(1,6)]
ko=['ko'+str(i3) for i3 in range(1,6)]
data=pd.DataFrame(columns=[*wt,*ko],index=genes)
for gene in data.index:
    data.loc[gene,'wt1':'wt5']=np.random.poisson(lam=rd.randrange(10,1000),size=5)
    data.loc[gene,'ko1':'ko5']=np.random.poisson(lam=rd.randrange(10,1000),size=5)
print(data.head())

#2-reduce difference between values in each row
scaled_data=preprocessing.scale(data.T)
pca =PCA()
pca.fit(scaled_data)
pca_data=pca.transform(scaled_data)


#3-show precentage variance plot for pc(about 10 genes)
per_var=np.round(pca.explanied_variance_ratio_*100,decimals=1)
labels=['PC'+str(i4) for i4 in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1),height=per_var)
plt.xlabel('Principal Component')
plt.ylabel('Percentage of Explained Variance')
plt.title('Scree plot')
plt.show()

#4-
pca_df=pd.DataFrame(pca_data,index=[*wt,*ko],columns=labels)
plt.scatter(pca_df.PC1,pca_df.PC2)
plt.title('PCA Graph')
plt.xlabel('PC1'+per_var[0])
plt.ylabel('PC2'+per_var[1])
for sample in pca_df.index:
    plt.annotate(sample , pca_df.PC1.loc[sample] , pca_df.PC2.loc[sample])
plt.show()
