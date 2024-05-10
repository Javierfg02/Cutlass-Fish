import tensorflow as tf
from decoders import Decoder, TransformerDecoder
from embeddings import Embeddings

def greedy(
        src_mask,
        embed,
        decoder,
        encoder_output,
        trg_input,
        model,
    ):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param embed: target embedding
    :param bos_index: index of <s> in the vocabulary
    :param max_output_length: maximum length for the hypotheses
    :param decoder: decoder to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder final state (unused in Transformer)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    # Extract just the BOS first frame from the target
    ys = tf.slice(trg_input, [0, 0, 0], [-1, 1, -1])

    # If the counter is coming into the decoder or not
    ys_out = ys

    # Set the target mask, by finding the padded rows
    trg_mask = tf.not_equal(trg_input, 0.0)
    trg_mask = tf.expand_dims(trg_mask, axis=1)

    # Find the maximum output length for this batch
    max_output_length = tf.shape(trg_input)[1]

    # If just count in, input is just the counter
    if model.just_count_in:
        ys = tf.slice(ys, [0, 0, -1], [-1, -1, 1])

    for i in range(max_output_length):

        # ys here is the input
        # Drive the timing by giving the GT timing - add in the counter to the last column

        if model.just_count_in:
            # If just counter, drive the input using the GT counter
            ys[:, -1, -1:] = tf.slice(trg_input, [0, i, -1], [-1, 1, -1])

        else:
            # Give the GT counter for timing, to drive the timing
            ys[:, -1, -1:] = tf.slice(trg_input, [0, i, -1], [-1, 1, -1])

        # Embed the target input before passing to the decoder
        trg_embed = embed(ys)

        # Cut padding mask to required size (of the size of the input)
        padding_mask = trg_mask[:, :, :i+1, :i+1]
        #padding_mask = None

        # Pad the mask (If required) (To make it square, and used later on correctly)
        #pad_amount = tf.shape(padding_mask)[2] - tf.shape(padding_mask)[3]
        pad_amount = None
        #padding_mask = tf.equal(tf.pad(padding_mask, [[0, 0], [0, 0], [pad_amount, 0], [0, 0]], mode='SYMMETRIC'), 1)
        #padding_mask = None

        # Pass the embedded input and the encoder output into the decoder
        out, _, _, _ = decoder(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            src_mask=src_mask,
            trg_mask=padding_mask,
        )

        if model.future_prediction != 0:
            # Cut to only the first frame prediction
            out = tf.concat([out[:, :, :out.shape[2] // (model.future_prediction)], out[:, :, -1:]], axis=2)

        if model.just_count_in:
            # If just counter in trg_input, concatenate counters of output
            ys = tf.concat([ys, tf.slice(out, [0, -1, -1], [-1, 1, -1])], axis=1)

        # Add this frame prediction to the overall prediction
        ys = tf.concat([ys, tf.slice(out, [0, -1, 0], [-1, 1, -1])], axis=1)

        # Add this next predicted frame to the full frame output
        ys_out = tf.concat([ys_out, tf.slice(out, [0, -1, 0], [-1, 1, -1])], axis=1)

    return ys_out, None