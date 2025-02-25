import numpy as np

#Define Queries , Keys and Values

queries = np.array([[1,0,1], [0,1,1]])
keys = np.array([[1,0,1], [1,1,0],[0,1,1]])
values = np.array([[10,0], [0,10],[5,5]])

#Compute Attention Score

scores = np.dot(queries,keys.T)


#Apply Softmax formula to normalize score

def softmax(x):
    exp_x = np.exp(x-np.max(x,axis=1,keepdims=True))
    return exp_x/exp_x.sum(axis=1,keepdims= True)


attention_weights =softmax(scores)

#Compute the weight sum of the values

context = np.dot(attention_weights,values)

print("Attention Weights : \n ", attention_weights)
print("Context  Vectors : \n ",context)