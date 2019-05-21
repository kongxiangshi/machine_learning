# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns

data = sns.load_dataset('iris')
print(data.columns)

plt.figure()
sns.pairplot(data, hue='species', diag_kind='hist', height=1.5)
plt.show()