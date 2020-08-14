import sys
sys.path.append("../")

import numpy as np
import pandas as pd

from skimage import io

# Helper functions to help others repeat and build off our results

# MLC labels when downloaded from original source are messy
# this cleans them up 
MLC_labels = pd.read_csv("../MLC_labels.csv")
MLC_labels = dict(zip(MLC_labels['Old'], MLC_labels['New']))


def get_sparse_points(image_file, label_file, labels, confidence = .9):
    '''
    Cleans original CPCe annotations, and concatenates the additional labels
    that were provided by the CNN (not included in this repo)
    '''
     
    # Read and clean the file, replace old labels
    original_sparse = pd.read_csv(label_file, sep = "; ", 
                                  engine = 'python').rename(columns={'# Row' : 'Y', 'Col': 'X'})
    
    original_sparse.replace(MLC_labels, inplace = True)
    
    # Remove the original Off points because they are inconsistant, then get good consistant replacements
    original_sparse = original_sparse[original_sparse['Label'] != 'Off']
    original_sparse = original_sparse.append(create_off_points(image_file, offset = 25, density = 10), ignore_index = True)
    original_sparse['Confidence'] = 1.0
    
    additional_sparse = pd.read_csv(label_file.replace("labels",
                                                       "additional_labels").replace(".jpg.txt", ".csv"))
    
    additional_sparse.drop(['Unnamed: 0'], axis = 1, inplace = True) 
    
    # There is merit in not trusting all of labels provided by the cnn
    # One safeguard (not perfect though), is to remove any additional 
    # labels for class categories that are not already present in the 
    # original annotations.
    if(True):
        labels_in_original = np.unique(original_sparse['Label'].values)
        
        temp_a = pd.DataFrame()
        for class_label in labels_in_original:
            temp_b = additional_sparse[additional_sparse['Label'] == class_label]
            temp_a = pd.concat([temp_a, temp_b])

        additional_sparse = temp_a
    
    # The second safeguard would be to remove all labels that are less likely to
    # be correct, der.
    combined_sparse = pd.concat([original_sparse, additional_sparse])
    combined_sparse = combined_sparse[combined_sparse['Confidence'] >= confidence]
    
    # Removes all of the other classes not apart of the MLC experiments
    temp_a = pd.DataFrame()
    for class_label in labels:
        temp_b = combined_sparse[combined_sparse['Label'] == class_label]
        temp_a = pd.concat([temp_a, temp_b])
    
    return temp_a

def create_off_points(image_file, offset = 25, density = 25):
    '''
    We found that it is pretty easy to just label the metal quadrat this way
    Simply puts 'Off' points along the edges and sends those labels back
    to be concatenated with the rest of the labels
    '''
       
    img = io.imread(image_file)
    
    x = []
    y = []
    l = []
    
    # Top
    x_ = [_ for _ in range(offset, img.shape[1] - offset, density)]
    y_ = [offset] * len(x_)
    
    x += x_
    y += y_
    
    # Bottom
    x_ = [_ for _ in range(offset, img.shape[1] - offset, density)]
    y_ = [img.shape[0] - offset] * len(x_)
    
    x += x_
    y += y_
    
    # Left
    y_ = [_ for _ in range(offset, img.shape[0] - offset, density)]
    x_ = [offset] * len(y_)
    
    x += x_
    y += y_
    
    # Right
    y_ = [_ for _ in range(offset, img.shape[0] - offset, density)]
    x_ = [img.shape[1] - offset] * len(y_)
    
    x += x_
    y += y_
    
    l = ['Off'] * len(x)
        
    return pd.DataFrame(data = list(zip(x, y, l)), columns = ['X', 'Y', 'Label'])