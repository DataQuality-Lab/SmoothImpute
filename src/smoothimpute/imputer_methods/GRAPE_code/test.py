import numpy as np
import torch

nrow = 5
ncol = 3
feature_ind = np.array(range(ncol))
feature_node = np.zeros((ncol,ncol))
feature_node[np.arange(ncol), feature_ind] = 1
sample_node = [[1]*ncol for i in range(nrow)]
node = sample_node + feature_node.tolist()

print(node)

print(torch.FloatTensor(10, 1).uniform_())

impute_hiddens = list(map(int,"64".split('_')))
print(impute_hiddens)