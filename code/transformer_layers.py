import math
import tensorflow as tf
import numpy as np 


class MultiHeadedAttention(tf.keras.Model):

    def __init__(self,num_heads,size,dropout):

        super().__init__()

        assert size % num_heads == 0
        self.head_size = size // num_heads
        self.model_size =size
        self.num_heads = num_heads

        # linear layers to extract the key, query and value vectors for the attention mechanism 
        self.k_layer = tf.keras.layers.Dense(num_heads*self.head_size)
        self.v_layer = tf.keras.layers.Dense(num_heads*self.head_size)
        self.q_layer = tf.keras.layers.Dense(num_heads*self.head_size)
        
        self.output = tf.keras.layers.Dense(size)
        self.softmax = tf.keras.layers.Softmax(dim=-1)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.target_pad = 0.0

    @tf.function
    def call(self,k,v,q,mask= None,padding_mask = None):

        batch_size = k.shape[0]

        # extracting the keys, queries and values 
        k = self.k_layer(k)
        q = self.q_layer(q)
        v = self.v_layer(v)

        #reshaping 
        k = tf.reshape(k,[batch_size,-1,self.num_heads,self.head_size]).transpose(1,2)
        q = tf.reshape(q,[batch_size,-1,self.num_heads,self.head_size]).transpose(1,2)
        v = tf.reshape(v,[batch_size,-1,self.num_heads,self.head_size]).transpose(1,2)

        #attention scores
        sim_score = tf.matmul(q,k.transpose(2,3))
        dim_k = np.sqrt(self.head_size)
        sim_score = sim_score/dim_k

        # masking future tokens in decoder self attention 
        if mask is not None:
            sim_score = tf.where(~mask, sim_score, float('-inf'))
        
        attention = self.softmax(sim_score)
        attention = self.droput(sim_score)

        if padding_mask is not None:
            attention = tf.where(~padding_mask,0.0)

        output = tf.matmul(attention,v)
        output = output.transpose(1,2).reshape(batch_size,-1,self.num_heads*self.head_size)
        
        output = self.output_layer(output)

        return output 

class PositionwiseFeedForward(tf.keras.Model):

    def __init__(self, input_size,ff_size,dropout=0.1):
        super().__init__()

        """
        This class handles layer norm, and residual stream which is used in the encoder and 
        decoder classes.
        """
        self.norm = tf.keras.layers.LayerNormalization()
        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_size,activation='ReLU'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(input_size),
            tf.keras.layers.Dropout(dropout)
        ])

    def call(self,attention_output):
        norm = self.norm(attention_output)
        return self.ff_layer(norm) + attention_output