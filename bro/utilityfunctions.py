import numpy as np
import pandas as pd
import pickle
import os
import shutil

def normalize_pdf(pdf):
    """
    Given PDF as numpy array where first column is values random variable can take and second column is proportional to the probability of those values, normalize the second column.
    """
    pdf[:,1] = pdf[:,1]/pdf[:,1].sum()
    return pdf 

def uniform_distribution_pdf(support):
    """
    Generates discrete uniform PDF on given support
    """
    return np.array([[val, 1/len(support)] for val in support])

def sample_from_distribution(dist, n=1):
    """Assumes dist has form dist=np.array([[val, prob of val],...]) and is numpy array
    """
    sample = np.random.choice(dist[:,0], size=n, replace=True, p=dist[:,1])
    return sample

def pickle_object(save_path, obj):
    """
    Pickle object to save_path location

    Parameters:
        -- save_path (str) : path to pickled object
        -- obj : whatever object you want pickled

    Returns:
        -- None
    """
    outfile = open(save_path, 'wb')
    
    
    pickle.dump(obj, outfile)
    outfile.close()

def unpickle_object(save_path):
    """
    Unpickle object saved to save_path location

    Parameters:
        -- save_path (str) : path to pickled object

    Returns:
        -- obj : pickled object
    """
    infile = open(save_path, 'rb')
    obj = pickle.load(infile)
    infile.close()

    return obj

def remove_and_make_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    return None

def flatten_list_of_lists(list_of_lists):
    """
    Given a list of lists, e.g., [[a,b], [c,d]], returns [a,b,c,d]
    """
    return [element for sublist in list_of_lists for element in sublist]

def print_from_dict(d, keys):
    out = [f'{key} = {d[key]}' for key in keys]
    print('; '.join(out))

def print_often(total_num, current_i, print_every=10):
    divisor = np.floor(total_num/print_every)
    if total_num<print_every: print(f'i = {current_i+1}/{total_num}')
    elif current_i%divisor==0: print(f'i = {current_i+1}/{total_num}')