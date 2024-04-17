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

def create_trg():
    json_path =  RAW_DATA_PATH + '/openpose_output/json/'
    directories = os.listdir(json_path) # get all the directories
    # for each directory get all of its json files (these store the joints data for each frame)
    json_data = None
    for i in directories:
        directory_path = os.path.join(json_path, i)
        files = os.listdir(directory_path)
        # each json file stores the data for a frame
        for j in sorted(files): # we need to maintain the sequence of frames so we sort files by name
            if(j == sorted(files)[0]):
                file_path = os.path.join(directory_path, j)
                # open the file
                with open(file_path, 'r') as file:
                    json_file = file.read()
                    json_data = json.loads(json_file)
                    # json_data = json_data['people'][0]['pose_keypoints_2d']
                    json_data = json_data['people']
    print(json_data)

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
    
    with open("../data/val/processed/sentence_name_to_sentence.json", "w") as outfile: 
        json.dump(src_dictionary, outfile)

    '''
    # Access a specific column by its name
    sentence_name = df['SENTENCE_NAME']
    sentence = df['SENTENCE']'''

    

if __name__ == '__main__':
    create_trg()
    # create_src()