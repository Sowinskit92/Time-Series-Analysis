# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:08:56 2025

@author: Tim.Sowinski
"""

# helper packages
import random
import os
os.chdir("C:/Users/Tim.Sowinski/OneDrive - Cornwall Insight Ltd/Documents/Flex Stuff/Flex_Digitalisation/TSC testing/TSC example")
import numpy as np
import pandas as pd
import sys
#from functools import lru_cache
#from data_cache import pandas_cache - INSTALLING THIS PACKAGE BROKE THE ROCKET CLASSIFIER INSTALL

# ML packages
#import tensorflow as tf
import matplotlib.pyplot as plt
import sktime
from sktime.transformations.panel.rocket import Rocket
#import lightgbm as lgbm

sys.exit()
pd.set_option("display.max_columns", None)

# =============================================================================
#                                   NOTES
# 
# - Taken from this video: https://youtu.be/0c0YNWo9Xyg?si=jUYM9O4-1y7m2bgY
# - I have also copied what the video had line for line, regardless of if 
#   there's a better way of doing it
# 
# - She mentions this is a very simple model and that having 2 or more classifiers 
#   and using 'ensambles' of them would be more accurate 
# 
# =============================================================================

# =============================================================================
#                            BACKGROUND TO DATA
# 
# - The data is readings from different sensors placed on people during 
#   one of two activities ('states') for 1 minute 
# - Readings were taken each second from each sensor over a minute to make up
#   a 'sequence'
# - There were many different people that took part (labelled as 'subject')
# - The goal of this is to predict the state variable of the testing set
# =============================================================================






# =============================================================================
# Loading data
# =============================================================================
# training data
df = pd.read_pickle("train.pickle")
# testing data
df_test = pd.read_pickle("test.pickle")
# label data
df_label = pd.read_csv("train_labels.csv")

# =============================================================================
# Examining training data
# =============================================================================

tot_sequences = len(df.sequence.unique())
tot_sequences_test = len(df_test.sequence.unique())
tot_subject = len(df.subject.unique())

print(f"There are {tot_sequences} sequences and {tot_subject} subjects")

df.set_index(["sequence", "subject"], drop = False, inplace = True)
df_test.set_index(["sequence", "subject"], drop = False, inplace = True)
df.drop("step", axis = 1, inplace = True)
df_test.drop("step", axis = 1, inplace = True)
#print(df.head(10))
print(f'We have {tot_sequences_test} sequences and {len(df_test.subject.unique())} subjects.')

# checks all sequences have 60s worth of data
# this is really slow so I'm commenting it out, but just know all the sequences have the correct amount of data
"""
complete = []
for i in range(tot_sequences):
    if df.loc[i].shape[0] == 60:
        complete.append(i)
    else:
        pass

# checks len of sequences with length 60 matches the number of sequences
print(len(complete))
"""
# =============================================================================
# Converting dataframes into 3d numpy matrices (Note: sktime now accepts 
# multi-indexed pandas dataframes, so this isn't a necessary step)
# =============================================================================

df.drop(["sequence", "subject"], axis = 1, inplace = True)
df_test.drop(["sequence", "subject"], axis = 1, inplace = True)


# checks to make sure the number of columns in the training and testing data are the same
print(df.shape[1] == df_test.shape[1]) 

#@functools.cache
#@lru_cache(maxsize = 256)
def get_3D(data, total_seq):
    # sktime requires the 3D data to be (total number of sequences, total variables, n_timepoints)
    
    # total variables = 13 as there's 13 sensors
    # n_timepoints = 60 as there's 60 readings per sequence
    new_data = np.zeros((total_seq, 13, 60))
    for i in range(total_seq):
        # .T transposes the rows
        new_data[i] = data.loc[i].T
    
    # a3d = np.array(list(pdf.groupby('a').apply(pd.DataFrame.as_matrix)))
    print(new_data.shape)
    return new_data

#@functools.cache
def get_3D_test(data, total_seq, total_seq_temp):
    new_data = np.zeros((total_seq_temp, 13, 60))
    #print(data)
    #print(total_seq)
    #print(total_seq_temp)
    
    for i in range(total_seq_temp):
        #print(i)
        #print(i + total_seq)
        # in the test dataset, the index starts at 25_968 which is the total_seq
        # total_seq in the train dataset (they could have used iloc but alas)
        new_data[i] = data.loc[i + total_seq].T
        '''
        try:
            new_data[i] = data.loc[i + total_seq].T
        except:
            print(i)
            print(i + total_seq)
            sys.exit()
        '''
    print(new_data.shape)
    return new_data
    
#print(df)
#print(df_test)

print("Transforming df to 3D array...")
#df_3d = get_3D(df, tot_sequences)
print("Transforming df_test to 3D array...")
#df_test_3d = get_3D_test(df_test, tot_sequences, tot_sequences_test)


# =============================================================================
# Splitting dataset into train/test/validation
# =============================================================================
rnd = np.random.default_rng(8)
print(rnd)
