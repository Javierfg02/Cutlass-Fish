import numpy as np
import math
import tensorflow as tf

from helpers import bpe_postprocess, load_config, get_latest_checkpoint, calculate_dtw
from preprocess import create_src, create_trg, create_examples, make_data_iter, map_src_sentences, pad_trg_data
from model import build_model
from batch import Batch
from preprocess import make_data_iter
from constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN

batch_size = 8


def preprocess_example(example):
    # Implement conversion of preprocess.Example to tensors
    # Example: Assuming `example` has attributes like src, trg, etc.
    src_tensor = tf.convert_to_tensor(example.src)  # Convert src to tensor
    trg_tensor = tf.convert_to_tensor(example.trg)  # Convert trg to tensor

    return (src_tensor, trg_tensor)  # Return tuple of tensors

def validate_on_data(model, data, batch_size, max_output_length, eval_metric, loss_function=None, batch_type="sentence", type="val", BT_model=None):
    # Preprocess data to create a list of TensorFlow-compatible tuples
    processed_data = [preprocess_example(example) for example in data]

    # Create TensorFlow dataset from processed_data
    valid_dataset = tf.data.Dataset.from_generator(lambda: processed_data, output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.int32),  # Example: src_tensor shape and dtype
        tf.TensorSpec(shape=(), dtype=tf.int32)   # Example: trg_tensor shape and dtype
    ))
    valid_dataset = valid_dataset.batch(batch_size)

# def validate_on_data(model,
#                      data,
#                      batch_size,
#                      max_output_length,
#                      eval_metric,
#                      loss_function=None,
#                      batch_type="sentence",
#                      type="val",
#                      BT_model=None):

#     # Create TensorFlow dataset and iterator
#     valid_dataset = tf.data.Dataset.from_tensor_slices(data)
    # valid_dataset = valid_dataset.batch(batch_size)
    # valid_iter = iter(valid_dataset) # TODO: NOT A FUNC?

    # def make_data_iter(dataset, shuffle=True, train=True):
    padded_trgs = pad_trg_data(sequences=[data[0].trg], batch_size=batch_size, num_features_per_frame=412)
    #print("padded_trgs", padded_trgs.shape)
    #print("td",train_data)
    valid_batch = map_src_sentences(dataset=data, padded_trgs=padded_trgs)

    valid_iter = make_data_iter(dataset=valid_batch, shuffle=True, train=False)

    # pad_index = model.src_vocab.index(PAD_TOKEN)
    pad_index = model.src_vocab.stoi[PAD_TOKEN]

    # model.eval()

    valid_hypotheses = []
    valid_references = []
    valid_inputs = []
    file_paths = []
    all_dtw_scores = []

    valid_loss = 0.0
    total_ntokens = 0
    total_nseqs = 0

    batches = 0

    
    for valid_batch in valid_iter:
        # Convert valid_batch to TensorFlow tensors
        # valid_batch = tf.convert_to_tensor(valid_batch)

        print(f"_VALID_BATCH: {valid_batch}")


        # Extract batch using Batch class (custom implementation for TensorFlow)
        batch = Batch(torch_batch=valid_batch,
                      pad_index=pad_index,
                      model=model)

        targets = batch.trg

        if loss_function is not None and targets is not None:
            # Get the loss for this batch
            batch_loss = model.get_loss_for_batch(batch, loss_function=loss_function)

            valid_loss += batch_loss.numpy()
            total_ntokens += batch.ntokens
            total_nseqs += batch.nseqs

        if not model.just_count_in:
            # Run batch through the model in an auto-regressive format
            output, attention_scores = model.run_batch(batch=batch,
                                                       max_output_length=max_output_length)

        if model.future_prediction != 0:
            # Cut to only the first frame prediction + add the counter
            output = tf.concat([output[:, :, :output.shape[2] // model.future_prediction],
                                output[:, :, -1:]], axis=2)

            targets = tf.concat([targets[:, :, :targets.shape[2] // model.future_prediction],
                                 targets[:, :, -1:]], axis=2)

        if model.just_count_in:
            output = output  # Assuming train_output is defined earlier

        # Convert TensorFlow tensors to numpy arrays for further processing
        valid_references.extend(targets.numpy())
        valid_hypotheses.extend(output.numpy())
        file_paths.extend(batch.file_paths.numpy())

        # Collect source sentences using model's source vocab and batch indices
        src_sentences = [[model.src_vocab[b].numpy().decode('utf-8') for b in batch.src[i].numpy()] 
                         for i in range(len(batch.src))]
        valid_inputs.extend(src_sentences)

        # Calculate Dynamic Time Warping score for evaluation
        dtw_score = calculate_dtw(targets.numpy(), output.numpy())
        all_dtw_scores.extend(dtw_score)

        # Limit the number of batches processed (for testing)
        if batches == math.ceil(20 / batch_size):
            break
        batches += 1

    # Calculate mean Dynamic Time Warping score
    current_valid_score = np.mean(all_dtw_scores)

    return current_valid_score, valid_loss, valid_references, valid_hypotheses, \
           valid_inputs, all_dtw_scores, file_paths
