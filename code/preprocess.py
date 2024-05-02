import sys
import os
import csv
import json
import numpy as np
import tensorflow as tf
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset
import pandas as pd
import re
from vocabulary import Vocabulary
import numpy as np
from random import shuffle
# import torchtext
from torchtext import data
from torchtext.data import Dataset, Iterator, Field
from tensorflow.keras.preprocessing.sequence import pad_sequences


RAW_DATA_PATH = '../data/val/raw'

def tokenize(text):
    """ 
    Use regex to separate words from punctuation and tokenize the text. 
    same method used in vocabulary
    """
    # Pattern to find words or punctuation
    text = text.lower() 
    pattern = re.compile(r"[\w']+|[.,!?;]")
    return pattern.findall(text)

def z_normalize(arr: np.ndarray) -> np.ndarray:
    '''
    Helper function to normalise an np.array using Z-score normalization
    '''
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    return (arr - mean_val) / std_val

def dim_normalize(arr, width=1280, height=720):
    '''
    Helper function to normalise an np.array using dimension normalization
    '''
    arr = np.array(arr, dtype=np.float64)
    arr[0::3] /= width  # x-coordinates
    arr[1::3] /= height # y-coordinates
    return arr

def min_max_normalize(arr):
    '''
    Helper function to normalise an np.array using min-max normalization
    '''
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr

def pos1_neg1_normalize(arr):
    arr = np.array(arr, dtype=np.float64) 
    min_val = np.min(arr)
    max_val = np.max(arr)
    # Normalize from -1 to 1
    return 2 * ((arr - min_val) / (max_val - min_val)) - 1 if max_val > min_val else arr

def create_trg():

    output = {}

    json_path =  RAW_DATA_PATH + '/openpose_output/json/'
    directories = os.listdir(json_path) # get all the directories
    # for each directory get all of its json files (these store the joints data for each frame)
    json_data = None

    for i in directories[:3]: # TODO CHANGE
        directory_path = os.path.join(json_path, i)
        files = os.listdir(directory_path)

        # each json file stores the data for a frame
        frames = []
        total_frames = len(files)
        for j, file_name in enumerate(sorted(files)): # we need to maintain the sequence of frames so we sort files by name

            file_path = os.path.join(directory_path, file_name)
            # open the file
            with open(file_path, 'r') as file:
                json_file = file.read()
                json_data = json.loads(json_file)
                # concatenate all keypoints
                # TODO: try different normalization functions
                pose_kp = pos1_neg1_normalize(np.array(json_data['people'][0]['pose_keypoints_2d']))
                hand_left_kp = pos1_neg1_normalize(np.array(json_data['people'][0]['hand_left_keypoints_2d']))
                hand_right_kp = pos1_neg1_normalize(np.array(json_data['people'][0]['hand_right_keypoints_2d']))
                face_kp = pos1_neg1_normalize(np.array(json_data['people'][0]['face_keypoints_2d']))
                
                concat_kp = np.concatenate((pose_kp, face_kp, hand_left_kp, hand_right_kp))

                #? Add a counter to delimit the ending of the sentence - ! Counter decoding technique
                normalized_counter = j / (total_frames - 1)  # Normalize the counter to be between 0 and 1, adjust for zero indexing
                frame_with_counter = np.append(concat_kp, normalized_counter)  # Append the normalized counter
                frames.append(frame_with_counter)

        if frames:  # Only process non-empty frame lists
            # Convert frame list to a stacked numpy array, then to a TensorFlow tensor
            # print("before frames: ", frames[0])
            frames_tensor = tf.convert_to_tensor(np.stack(frames, axis=0), dtype=tf.float32)
            # print("after frames: ", frames_tensor)
            # print("AFTER FRAMES: ",tf.expand_dims(frames_tensor, axis=0))


            # frames_tensor = tf.convert_to_tensor(np.stack(frames, axis=0), dtype=tf.float32)
            # output[i] = [frames_tensor]
            output[i] = frames_tensor
            # output[i] = tf.expand_dims(frames_tensor, axis=0)
 
    #? First directory name for testing: 279MO2nwC_E_8-2-rgb_front
    return output

def create_src():
    csv_path = RAW_DATA_PATH + "/how2sign_realigned_val.csv"
    src_dictionary = {}
    with open(csv_path, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        
        # Iterate through each row in the CSV file
        for row in reader:
            split_row = row[0].split('\t')
           
            sentence_name = split_row[3]
            sentence = split_row[6]
            
            src_dictionary[sentence_name] = sentence
            
    return src_dictionary

def make_data_iter(dataset, shuffle=True, train=True):
    """
    Shuffles dataset. That's all.
    """
    if shuffle and train:
        np.random.shuffle(dataset)
    return dataset

def map_src_sentences(dataset):
    """
    Converted Dataset to a list of Examples, but with src containing the mappings
    from words to indices in a given sentence
    
    Builds the 2 tensors in src fields for both correspondings with padding and their corresponding lengths before padding.
    """
    new_dataset = []
    tokenized_sentences = []
    len_unpadded_seq = []
    # First loop to get max length for all src fields in dataset (for padding)
    for example in dataset:
        indexed_tokens = example.src
        # print("INDEXED TOKENS: ", indexed_tokens)
        tokenized_sentences.append(indexed_tokens)
    
    for example in dataset:
        len_unpadded_seq.append(len(indexed_tokens))
        len_unpadded_seq_tensor = tf.convert_to_tensor(len_unpadded_seq, dtype=tf.int32)
        # print("len_unpadded_seq: ", len_unpadded_seq)
        # print("tokenized_sentences: ", tokenized_sentences)
        max_length = max(len(seq) for seq in tokenized_sentences)
        # Pad sequences to the maximum length
        padded_sequences = pad_sequences(tokenized_sentences, maxlen=max_length, padding='post')
        padded_sequences_tensor = tf.convert_to_tensor(padded_sequences, dtype=tf.int32)

        # print("Padded sequences tensor: ", padded_sequences_tensor)
        # new_src = padded_sequences_tensor, len_unpadded_seq_tensor
        new_src = padded_sequences_tensor
        # print("NEW SRC: ", new_src)
        new_example =  Example(
            src=new_src,
            trg=example.trg,
            file_path=example.file_path
        )
        new_dataset.append(new_example)

    return new_dataset

# def make_data_iter(dataset: Dataset,
#                    batch_size: int,
#                    batch_type: str = "sentence",
#                    train: bool = False,
#                    shuffle: bool = False) -> Iterator:
#     """
#     Returns a torchtext iterator for a torchtext dataset.

#     :param dataset: torchtext dataset containing src and optionally trg
#     :param batch_size: size of the batches the iterator prepares
#     :param batch_type: measure batch size by sentence count or by token count
#     :param train: whether it's training time, when turned off,
#         bucketing, sorting within batches and shuffling is disabled
#     :param shuffle: whether to shuffle the data before each epoch
#         (no effect if set to True for testing)
#     :return: torchtext iterator
#     """

#     batch_size_fn = None if batch_type == "token" else None

#     if train:
#         # optionally shuffle and sort during training
#         data_iter = data.BucketIterator(
#             repeat=False, sort=False, dataset=dataset,
#             batch_size=batch_size, batch_size_fn=batch_size_fn,
#             train=True, sort_within_batch=True,
#             sort_key=lambda x: len(x.src), shuffle=shuffle)
#     else:
#         # don't sort/shuffle for validation/inference
#         data_iter = data.BucketIterator(
#             repeat=False, dataset=dataset,
#             batch_size=batch_size, batch_size_fn=batch_size_fn,
#             train=False, sort=False)

#     return data_iter

        
class Example:
    def __init__(self, src, trg, file_path=None):
        self.src = src
        self.trg = trg
        self.file_path = file_path

def create_examples(src, trg):
    """
    Create a list of Example objects from source and target dictionaries
    
    Parameters:
        src: the source setnence
        trg: the target frame data

    Returns:
        A list of example objects
    """

    examples = []
    vocab = Vocabulary()
    vocab._from_file('../configs/src_vocab.txt')
    stoi = vocab.stoi
    # build vocab

    for key in src.keys():
        if key in trg:
            tokens = tokenize(src[key])
            indexed_tokens = [stoi.get(token, stoi['<unk>']) for token in tokens]
            
            example = Example(
                src=indexed_tokens, # convert words to their matching index
                trg=trg[key],
                file_path=key
            )
            examples.append(example)
        else:
            # print(f'Warning: Key {key} found in source but not in target')
            pass

    return examples


def test():
    trg_dict = create_trg() # shape = (x, 411 + counter) -> (num frames, data points per frame + counter)
    src_dict = create_src()
    examples = create_examples(src_dict, trg_dict)

    for example in examples[:5]:  # Check the first 5 examples
        counters = example.trg.numpy()[:, -1]  # Extract counter values from the last column
        # print(f"Counters for {example.file_path}: {counters}")

        # Assert the counters start at 0 and end at 1, and they are monotonically increasing
        assert counters[0] == 0, "Counter does not start at 0."
        assert counters[-1] == 1, "Counter does not end at 1."
        assert np.all(np.diff(counters) > 0), "Counters are not monotonically increasing." # we take the difference between consecutive
        # counters and if it is negative then they are not monotonically increasing

        # Print the first 5 frames' info for a quick check
        print(f"Source: {example.src}\n")
        print(f"Target: {example.trg}\n")
        print(f"Target shape: {example.trg.shape}\n")
        # print(f"Target type: {type(example.trg)}\n")
        print(f"File Path: {example.file_path}\n")

def main():
    test()

if __name__ == '__main__':
    main()