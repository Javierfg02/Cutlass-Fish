import tensorflow as tf
from constants import TARGET_PAD

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, model):

        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_lengths = torch_batch.src
        self.src_mask = tf.expand_dims(tf.not_equal(self.src, pad_index), axis=1)
        self.nseqs = tf.shape(self.src)[0]
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        self.file_paths = torch_batch.file_paths
        self.use_cuda = model.use_cuda
        self.target_pad = TARGET_PAD
        # Just Count
        self.just_count_in = model.just_count_in
        # Future Prediction
        self.future_prediction = model.future_prediction

        if hasattr(torch_batch, "trg"):
            trg = torch_batch.trg
            trg_lengths = trg.shape[1]
            # trg_input is used for teacher forcing, last one is cut off
            # Remove the last frame for target input, as inputs are only up to frame N-1
            self.trg_input = tf.identity(trg[:, :-1, :])

            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = tf.identity(trg[:, 1:, :])

            # Just Count
            if self.just_count_in:
                # If Just Count, cut off the first frame of trg_input
                self.trg_input = self.trg_input[:, :, -1:]

            # Future Prediction
            if self.future_prediction != 0:
                # Loop through the future prediction, concatenating the frames shifted across once each time
                future_trg = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
                # Concatenate each frame (Not counter)
                for i in range(0, self.future_prediction):
                    future_trg = future_trg.write(i, self.trg[:, i:-(self.future_prediction - i), :-1])
                # Create the final target using the collected future_trg and original trg
                self.trg = tf.concat([future_trg.concat(), self.trg[:, :-self.future_prediction, -1:]], axis=2)

                # Cut off the last N frames of the trg_input
                self.trg_input = self.trg_input[:, :-self.future_prediction, :]

            # Target Pad is dynamic, so we exclude the padded areas from the loss computation
            trg_mask = tf.expand_dims(tf.not_equal(self.trg_input, self.target_pad), axis=1)
            # This increases the shape of the target mask to be even (16,1,120,120) -
            # adding padding that replicates - so just continues the False's or True's
            pad_amount = tf.shape(self.trg_input)[1] - tf.shape(self.trg_input)[2]
            # Create the target mask the same size as target input
            self.trg_mask = tf.equal(tf.pad(trg_mask, [[0, 0], [0, 0], [pad_amount, 0], [0, 0]], mode='SYMMETRIC'), 1.0)
            self.ntokens = tf.cast(tf.reduce_sum(tf.cast(tf.not_equal(self.trg, pad_index), tf.float32)), tf.int32)

        if self.use_cuda:
            self._make_cuda()

    # If using Cuda
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