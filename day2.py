from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense,Add,Input,LayerNormalization,MultiHeadAttention
from tensorflow.keras.models import Model

#Define A simplified Transformer Encoder Block

def transformer_encoder(input_dim, num_heads,ff_dim):
    inputs=Input(shape=(None,input_dim))

    #Multi-head Self Attention
    attention_output=MultiHeadAttention(num_heads=num_heads,key_dim=input_dim)(inputs,inputs)
    attention_output=Add()([inputs, attention_output])
    attention_output=LayerNormalization()(attention_output)

    #FeedForward Neural network
    ff_output=Dense(ff_dim,activation='relu')(attention_output)
    ff_output=Dense(input_dim)(ff_output)
    outputs = Add()([attention_output,ff_output])
    outputs=LayerNormalization()(outputs)
    return Model(inputs,outputs)

#Create and Visualize  a Sample  Transformer Encoder Block

encoder_block=transformer_encoder(input_dim=64,num_heads=8,ff_dim=128)
plot_model(encoder_block,show_shapes=True,to_file="Transformer_encoder.png")