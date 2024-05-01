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
        
        self.output_layer = tf.keras.layers.Dense(size)
        self.softmax = tf.keras.layers.Softmax(axis=-1)
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
        attention = self.dropout(sim_score)

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
            tf.keras.layers.Dense(ff_size,activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(input_size),
            tf.keras.layers.Dropout(dropout)
        ])

    def call(self,attention_output):
        norm = self.norm(attention_output)
        return self.ff_layer(norm) + attention_output


# pylint: disable=arguments-differ
# class PositionalEncoding(tf.Module):

#     def __init__(self,
#                  size: int = 0,
#                  max_len: int = 200000, # Max length was too small for the required length
#                  mask_count=False):

#         if size % 2 != 0:
#             raise ValueError("Cannot use sin/cos positional encoding with "
#                              "odd dim (got dim={:d})".format(size))
#         # pe = tf.zeros(max_len, size)
#         pe = np.zeros((max_len, size))
#         # position = tf.range(0, max_len).unsqueeze(1)
#         position = np.arange(0, max_len).reshape(-1, 1)
#         div_term = tf.exp((tf.range(0, size, 2, dtype=tf.float32) *
#                               -(math.log(10000.0) / size)))
#         # pe[:, 0::2] = tf.sin(position.float() * div_term)
#         # pe[:, 1::2] = tf.cos(position.float() * div_term)
#         pe[:, 0::2] = np.sin(position * div_term)
#         pe[:, 1::2] = np.cos(position * div_term)
#         # pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
#         pe = pe[np.newaxis, :, :]  # shape: [1, max_len, size]

#         super(PositionalEncoding, self).__init__()
#         # self.register_buffer('pe', pe)
#         self.pe = tf.constant(pe, dtype=tf.float32)
#         self.dim = size
#         self.mask_count = mask_count

#     def forward(self, emb):

#         return emb + self.pe[:, :emb.size(1)]

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, size: int, max_len: int = 2000):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(size, max_len)
        
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, d_model, position):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        print("X: ", x)
        seq_len = x.shape[1]
        print("self.pos_encoding: ", self.pos_encoding)
        return x + self.pos_encoding[:, :seq_len]


class TransformerEncoderLayer(tf.keras.Model):

    def __init__(self,
                 size: int = 0,
                 ff_size: int = 0,
                 num_heads: int = 0,
                 dropout: float = 0.1):

        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size,
                                                dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.size = size

    def call(self, x, mask):

        x_norm = self.layer_norm(x)

        h = self.src_src_att(x_norm, x_norm, x_norm, mask=mask)

        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o
    
class TransformerDecoderLayer(tf.keras.Model):

    def __init__(self,
                 size: int = 0,
                 ff_size: int = 0,
                 num_heads: int = 0,
                 dropout: float = 0.1,
                 decoder_trg_trg: bool = True):

        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size,
                                                dropout=dropout)

        self.src_trg_att = MultiHeadedAttention(num_heads, size,
                                                dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)

        self.x_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dec_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.decoder_trg_trg = decoder_trg_trg

    def call(self,
                x,
                memory,
                src_mask,
                trg_mask,
                padding_mask):

        # decoder/target self-attention
        h1 = self.x_layer_norm(x)

        # Target-Target Self Attention
        if self.decoder_trg_trg:
            h1 = self.trg_trg_att(h1, h1, h1, mask=trg_mask, padding_mask=padding_mask)
        h1 = self.dropout(h1) + x

        # Source-Target Self Attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o
