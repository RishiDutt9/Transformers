import torch 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#Define Queries , Keys and Values

queries = torch.tensor([[1.0,0.0,1.0], [0.0,1.0,1.0]])
keys = torch.tensor([[1.0,0.0,1.0], [1.0,1.0,0.0],[0.0,1.0,1.0]])
values = torch.tensor([[10.0,0.0], [0.0,10.0],[5.0,5.0]])

#Compute  Attention scores

scores = torch.matmul(queries, keys.T)

#Apply softmax to Normalize the Scores

attention_weights = F.softmax(scores,dim=-1)

#Compute Weighted sum of values

context = np.dot(attention_weights,values)

print("Attention Weights :\n",attention_weights)
print("Context Vectors : \n ", context)


plt.matshow(attention_weights)
plt.colorbar()
plt.title("Attention weights ")
plt.show()