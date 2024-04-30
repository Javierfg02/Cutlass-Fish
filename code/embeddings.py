import tensorflow as tf

class MaskedNorm(tf.keras.layers.Layer):
    """
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, norm_type, num_groups, num_features):
        super(MaskedNorm, self).__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norm = tf.keras.layers.BatchNormalization(axis=-1)
        elif self.norm_type == "group":
            self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        elif self.norm_type == "layer":
            self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def call(self, x, mask):
        if self.training:
            reshaped = tf.reshape(x, [-1, self.num_features])
            reshaped_mask = tf.reshape(mask, [-1, 1]) > 0
            selected = tf.boolean_mask(reshaped, reshaped_mask)
            selected = tf.reshape(selected, [-1, self.num_features])
            batch_normed = self.norm(selected)
            scattered = tf.where(reshaped_mask, batch_normed, reshaped)
            return tf.reshape(scattered, [tf.shape(x)[0], -1, self.num_features])
        else:
            reshaped = tf.reshape(x, [-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return tf.reshape(batched_normed, [tf.shape(x)[0], -1, self.num_features])

class Embeddings(tf.keras.layers.Layer):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 embedding_dim=64,
                 scale=False,
                 vocab_size=0,
                 padding_idx=1,
                 freeze=False,
                 **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                             mask_zero=True, input_length=None)

        if freeze:
            # freeze_params(self)
            self.lut.trainable = False

    # pylint: disable=arguments-differ
    def call(self, x):
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * tf.sqrt(float(self.embedding_dim))
        return self.lut(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)
