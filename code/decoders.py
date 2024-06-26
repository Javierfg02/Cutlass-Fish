# TRANSLATE TO TF 

import tensorflow as tf

# from helpers import freeze_params, ConfigurationError, subsequent_mask, uneven_subsequent_mask
from helpers import freeze_params, ConfigurationError, subsequent_mask
from transformer_layers import PositionalEncoding, \
    TransformerDecoderLayer

class Decoder(tf.keras.Model):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self._output_size

class TransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 1,
                 freeze: bool = False,
                 trg_size: int = 97,
                 decoder_trg_trg_: bool = True,
                 **kwargs):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(TransformerDecoder, self).__init__()

        self._hidden_size = hidden_size

        # Dynamic output size depending on the target size
        self._output_size = trg_size

        # create num_layers decoder layers and put them in a list
    
        # self.decoder_layers = tf.keras.Sequential([TransformerDecoderLayer(hidden_size,
                                                                #    ff_size,num_heads,dropout,decoder_trg_trg_) for _ in range (num_layers)])
        self.decoder_layers =[TransformerDecoderLayer(hidden_size,
                                                                   ff_size,num_heads,dropout,decoder_trg_trg_) for _ in range (num_layers)]

        self.pe = PositionalEncoding(hidden_size)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.emb_dropout = tf.keras.layers.Dropout(emb_dropout)

        # Output layer to be the size of joints vector + 1 for counter (total is trg_size)
        self.output_layer = tf.keras.layers.Dense(trg_size, use_bias=False)
        if freeze:
            freeze_params(self)

    def call(self,trg_embed, encoder_output, src_mask, trg_mask,
                **kwargs):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        # add position encoding to word embedding
        x = self.pe(trg_embed)
        # Dropout if given
        x = self.emb_dropout(x)

 
        padding_mask = trg_mask
        #print("SUB MASK: ", trg_mask.shape)
        #padding_mask = None
        # Create subsequent mask for decoding
        trg_size = trg_embed.shape[1]  # Get the size from the second dimension of trg_embed
        sub_mask = subsequent_mask(trg_size)
        sub_mask = tf.cast(sub_mask, dtype=trg_mask.dtype)
        # sub_mask = subsequent_mask(
            # trg_embed.shape[1]).type_as(trg_mask)

            # trg_embed.size(1)).type_as(trg_mask)

        # Apply each layer to the input
        for layer in self.decoder_layers:
            x = layer(x=x, memory=encoder_output,
                      src_mask=src_mask, trg_mask=sub_mask, padding_mask=padding_mask)

        # Apply a layer normalisation
        x = self.layer_norm(x)
        # Output layer turns it back into vectors of size trg_size
        output = self.output_layer(x)

        return output, x, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)
