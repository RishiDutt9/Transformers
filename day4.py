import numpy as np
import matplotlib.pyplot as plt


#Define Positional encoding function
def positional_encoding(seq_len,embed_dim):
    pos=np.arange(seq_len)[:,np.newaxis]
    i = np.arange(embed_dim)[np.newaxis,:]
    angle_rates= 1 /np.power(10000,(2*(i // 2))/np.float32(embed_dim))
    angle_rads = pos * angle_rates

    #Apply Sine to even indices and Cosines to Odd Indecies

    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:,0::2]=np.sin(angle_rads[:,0::2])
    pos_encoding[:,1::2]=np.cos(angle_rads[:,1::2])
    return pos_encoding


#Generate position encoding
seq_len  = 50
embed_dim = 16
pos_encoding =positional_encoding(seq_len=seq_len,embed_dim=embed_dim)

#Visualize the Positional Encoding

plt.figure(figsize=(10,6))
plt.pcolormesh(pos_encoding,cmap='viridis')
plt.colorbar()
plt.title("Positional Encoding")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Position")
plt.show()

#With  Pytorch

import torch 
import torch.nn as nn


class TransformerWithpositionalEncoding(nn.Module):
    def __init__(self,embed_dim,seq_dim, num_heads,ff_dim):
        super(TransformerWithpositionalEncoding,self).__init__()
        self.embedding = nn.Embedding(seq_dim,embed_dim)
        self.positional_encoding = nn.Parameter(torch.tensor(positional_encoding(seq_dim,embed_dim),dtype=torch.float32))
        self.multihead_attention = nn.MultiheadAttention(embed_dim,num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim,ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self,x):
        #Add positional encoding to embedding 
        x = self.embedding(x) + self.positional_encoding

        #self Attention

        attn_output, _ = self.multihead_attention(x,x,x)
        x = self.norm1(x+attn_output)

        #Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x+ffn_output)
        return x


#Define the MOdel Parameters

embed_dim = 16
seq_len = 50
num_heads = 4
ff_dim = 64

model=TransformerWithpositionalEncoding(embed_dim,seq_len,num_heads,ff_dim)

print(model)

#Learnable Positional Encoding

class LearnablePositionalEncoding(nn.Module):
    def __init__(self,embed_dim,seq_dim):
        super(LearnablePositionalEncoding,self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(seq_dim,embed_dim))

    def forward(self,x):
        return x + self.positional_encoding

learnable_pe = LearnablePositionalEncoding(seq_dim=seq_len,embed_dim=embed_dim)

print(learnable_pe)