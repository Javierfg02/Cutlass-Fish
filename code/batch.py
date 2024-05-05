import tensorflow as tf
from constants import TARGET_PAD
from preprocess import adjust_mask

class Batch:
    """
    Object for holding a batch of data with mask during training.
    Input is a batch from text iterator.
    """

    def __init__(self, torch_batch, pad_index, model):
        """
        Create a new batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths = torch_batch.src, len(torch_batch.src)
        # print("torch batch src: ", self.src)
        self.src_mask = tf.expand_dims(tf.not_equal(self.src, pad_index), axis=1)
        self.nseqs = tf.shape(self.src)[0]
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        self.use_cuda = model.use_cuda
        self.target_pad = TARGET_PAD
        self.just_count_in = model.just_count_in
        self.future_prediction = model.future_prediction

        if hasattr(torch_batch, "trg"):
            trg = torch_batch.trg
            trg_lengths = trg.shape[1]
            self.trg_input = tf.identity(trg[:, :-1])
            self.trg_lengths = trg_lengths
            self.trg = tf.identity(trg[:, 1:])

            if self.just_count_in:
                self.trg_input = self.trg_input[:, :, -1:]

            if self.future_prediction != 0:
                future_trg = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
                for i in range(0, self.future_prediction):
                    future_trg = future_trg.write(i, self.trg[:, i:-(self.future_prediction - i), :-1])
                self.trg = tf.concat([future_trg.concat(), self.trg[:, :-self.future_prediction, -1:]], axis=2)
                self.trg_input = self.trg_input[:, :-self.future_prediction, :]
            print("TARGET INPUT: ", self.trg_input)

            # print("TARGET INPUT: ", self.trg_input.shape)
            trg_mask = tf.expand_dims(self.trg_input != self.target_pad, axis=1)
            print("MASKKKKKK: ", trg_mask.shape)
    
            # trg_mask = tf.expand_dims(tf.not_equal(self.trg_input, self.target_pad), axis=1)
            # pad_amount = tf.maximum(0, tf.shape(self.trg_input)[1] - tf.shape(trg_mask)[2])
            #pad_amount = tf.shape(self.trg_input)[1] - tf.shape(trg_mask)[2]
            pad_amount =  self.trg_input.shape[1] - self.trg_input.shape[2]
            print('pad amounttt', pad_amount)
            if pad_amount > 0:
                self.trg_mask = tf.equal(tf.pad(trg_mask, [[0, 0], [0, 0], [0, pad_amount], [0, 0]], mode='CONSTANT', constant_values=False), True)
            else:
                self.trg_mask = trg_mask
            self.ntokens = tf.cast(tf.reduce_sum(tf.cast(tf.not_equal(self.trg, pad_index), tf.float32)), tf.int32)

        if self.use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = tf.identity(self.src)
        self.src_mask = tf.identity(self.src_mask)
        if self.trg_input is not None:
            self.trg_input = tf.identity(self.trg_input)
            self.trg = tf.identity(self.trg)
            self.trg_mask = tf.identity(self.trg_mask)
