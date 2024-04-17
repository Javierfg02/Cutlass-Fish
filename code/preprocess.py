import sys
import os
import csv
import json
import numpy as np
import tensorflow as tf
import pandas as pd

# TODO: Join json files for each sentence
# TODO: Investigate, are the joint values different in each frame??
# TODO: Add counter after each n joint values (n is the number of joint values per frame)
# TODO: Make src and vocab
# TODO: Make target

RAW_DATA_PATH = '../data/val/raw'

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

def pos1_neg1_normalize(arr, ):
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
    for i in directories:
        directory_path = os.path.join(json_path, i)
        files = os.listdir(directory_path)

        if (i != directories[0]): #! Added to save time. Only goes through the first directory - need to go through all in the end
            continue
        # each json file stores the data for a frame
        frames = []
        for j in sorted(files): # we need to maintain the sequence of frames so we sort files by name

            file_path = os.path.join(directory_path, j)
            # open the file
            with open(file_path, 'r') as file:
                json_file = file.read()
                json_data = json.loads(json_file)
                # concatenate all keypoints
                # TODO: try different normalization functions
                pose_kp = (np.array(json_data['people'][0]['pose_keypoints_2d']))
                hand_left_kp = (np.array(json_data['people'][0]['hand_left_keypoints_2d']))
                hand_right_kp = (np.array(json_data['people'][0]['hand_right_keypoints_2d']))
                face_kp = (np.array(json_data['people'][0]['face_keypoints_2d']))
                
                concat_kp = np.concatenate((pose_kp, face_kp, hand_left_kp, hand_right_kp))
                frames.append(concat_kp)

        # create a dictionary mapping directory_name (sentence_ID) to a 2D array where each row is a frame
        #? Do we want each row to be a frame?
        output[i] = np.stack(frames, axis=0)
    # concatenate all sequences from each file.
    #? First directory name for testing: 279MO2nwC_E_8-2-rgb_front
    print(output) # TODO: Once we have the mapping from directory_name to frame data, we need to join this with the source file to
    # TODO: to get a mapping of sentence to frame data
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
            print(split_row)
           
            sentence_name = split_row[3]
            sentence = split_row[6]
            
            src_dictionary[sentence_name] = sentence
            
    
    # with open("../data/val/processed/sentence_name_to_sentence.json", "w") as outfile: 
    #     json.dump(src_dictionary, outfile)
    return src_dictionary

    '''
    # Access a specific column by its name
    sentence_name = df['SENTENCE_NAME']
    sentence = df['SENTENCE']'''

if __name__ == '__main__':
    create_trg()
    # create_src()