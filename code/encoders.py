import tensorflow as tf
from transformer_layers import \
    TransformerEncoderLayer, PositionalEncoding
from helpers import freeze_params

class Encoder(tf.keras.Model):
    def output_size(self):
        return self._output_size
class TransformerEncoder(tf.keras.Model):
    """
    Encoder class 
    """

    # def __init__(self,hidden_size,ff_size,layers,heads,dropout,emb_dropout,freeze,**kwargs):
    def __init__(self,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the Encoder .
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: boolean to freeze the parameters of the encoder during training
        :param kwargs:
        """
        # super().__init__(**kwargs)
        super(TransformerEncoder, self).__init__()
        # self.encoder_layers = tf.keras.Sequential([TransformerEncoderLayer(hidden_size,ff_size,num_heads,dropout) for _ in range (num_layers)])
        self.encoder_layers = [TransformerEncoderLayer(hidden_size,ff_size,num_heads,dropout) for _ in range (num_layers)]
        self.norm = tf.keras.layers.LayerNormalization()
        self.pos_encoding = PositionalEncoding(hidden_size)
        self.emb_dropout = tf.keras.layers.Dropout(emb_dropout)
        
        if freeze:
            freeze_params(self)

    def call(self,embed,window_size,padding):
        """
        Passes the input into the encoder layers 

        param:
        embed: embedded text inputs, shape (batch_size, window_size, embed_size)
        window_size: length of text inputs before padding, shape (batch_size)
        padding: to allow padding in order for all inputs to have same window_size,
        shape (batch_size,window_size,embed_size)

        return:
        - output: hidden states with shape (batch_size,max_length,num_heads*hidden)
        """
        # adding positinal encoding
        inputs = self.pos_encoding(embed)
        # adding dropout
        inputs = self.emb_dropout(inputs)

        for layer in self.encoder_layers:
            inputs = layer(inputs, padding)

        norm = self.norm(inputs)
        return norm, None