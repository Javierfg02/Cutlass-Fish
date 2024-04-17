import sys
import os
import json
import numpy as np
import tensorflow as tf

# TODO: Join json files for each sentence
# TODO: Investigate, are the joint values different in each frame??
# TODO: Add counter after each n joint values (n is the number of joint values per frame)
# TODO: Make src 

def load_data():
    '''
    Loads the data
    '''
