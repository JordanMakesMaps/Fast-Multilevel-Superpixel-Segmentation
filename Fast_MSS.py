import gc

import cv2
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from skimage import io

from fast_slic.avx2 import SlicAvx2

# universal value denoting no label
NO_LABEL = 255

def synthesize_sparse_labels(gt, labels, percent, ratio, offset = 15):
 
    '''
    NOTE Only used in benchmarking

    When provided with dense annotations for an image, the function
    returns a pandas dataframe containing synthesized sparse annotations.

    gt      -> ground-truth dense annotations in a numpy array (W, H, 1)
    labels  -> all class labels that could be present in dataset
    percent -> percentage of sparse labels that should be sampled, (ie .001 == .1%)
    ratio   -> determines the % of points that are uniformly (1) or randomly (0) sampled
    offset  -> how many pixels away from the perimeter sampling occurs
    '''
    # total number of labels sampled, 
    # how closely they should be sampled (only applied to uniform)
    num_points = int((gt.shape[0] * gt.shape[1]) * percent)
    density = int(np.sqrt(num_points)) 

    # creates an equally spaced grid, reshapes, converts into list
    x_, y_ = np.meshgrid(np.linspace(offset, gt.shape[1] - offset, int(density * ratio)), 
                         np.linspace(offset, gt.shape[0] - offset, int(density * ratio)))

    xy = np.dstack([x_, y_]).reshape(-1, 2).astype(int)

    x = [point[0] for point in xy]
    y = [point[1] for point in xy]

    # Any labels that did not fit in the grid will be sampled randomly
    x += np.random.randint(offset, gt.shape[1] - offset, num_points - len(xy)).tolist()
    y += np.random.randint(offset, gt.shape[0] - offset, num_points - len(xy)).tolist()
    
    # extacts the labels from gt, stores in numpy array
    sparse = np.full(shape = gt.shape, fill_value = NO_LABEL, dtype = 'uint8')
    sparse[y, x] = gt[y, x]

    # creates a list containing all labels
    l = [sparse[point[1], point[0]] for point in list(zip(x, y))]
    l = [list(labels)[point] for point in l]

    # returns a pandas dataframe
    return pd.DataFrame(data = list(zip(x, y, l)), columns = ['X', 'Y', 'Label'])

def colorize_prediction(mask, labels):

    '''
    Simple function to colorize the mask
    mask   -> the mask created by fast_mss
    labels -> all possible labels in dataset
    '''
    # Creates a color palette
    cp = sns.color_palette("hls", len(labels))
   
    colored_mask = np.zeros(shape = (mask.shape[0], mask.shape[1], 3))

    # recolors all labels 
    for _ in np.unique(mask):
           
            colored_mask[mask == _] = cp[_]
        
    # returns a numpy array, (H, W, 3), [0, 1]
    return colored_mask

def display(a, b):

    '''
    A simple display function that shows the image and mask
    a & b  -> the actual image, the ground-truth dense annotations or mask
    '''

    plt.figure(figsize = (10, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(a)
    
    plt.subplot(1, 2, 2)
    plt.imshow(b)

    plt.show()
    
def compute_segmentations(start_iter, end_iter, num_iter):
    '''
    Copied from Alonso et al. 2019 (CoralSeg)
    This function calculated how many superpixels should be formed during each
    iteration of Fast-MSS, and is dependent on parameters required from the user
    '''
    # value for determining how much to decrease by each iteration
    reduction_factor = math.pow(float(end_iter) / start_iter, 1. / (num_iter - 1))

    # returns a list containing values in descending order
    return [int(start_iter * math.pow(reduction_factor, iters)) for iters in np.arange(num_iter)]

def mode(a, NO_LABEL = 255, axis = 0):

    '''
    Scipy's code to calculate the statistical mode of an array
    Here we include the ability to input a NULL value that should be ignored
    So in no situation should an index in the resulting dense annotations contain
    the background/NULL value.
    '''

    a = np.array(a)
    scores = np.unique(np.ravel(a))  
    testshape = list(a.shape)       
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape, dtype = int)    
    oldcounts = np.zeros(testshape, dtype = int)       

    for score in scores:

        # if the mode is a null value,
        # use the second most common value instead
        if(score == NO_LABEL):
            continue

        template = (a == score)                                 
        counts = np.expand_dims(np.sum(template, axis), axis)

        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)  
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent[0]

def fast_mss(image, sparse, labels, start_iter, end_iter, num_iter, method = 'mode'):
    
    all_masks = []
    segmentations = compute_segmentations(start_iter, end_iter, num_iter)

    # Loops through, each time a mask is created with different settings, and is accumulated
    for _ in range(num_iter):
        
        n_segments = segmentations[_]
    
        # Uses CPU to create segmented image with current params
        slic = SlicAvx2(num_components = n_segments, compactness = 25)
        segmented_image = slic.iterate(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))

        # The XY location of each annotation, along with the label
        X = np.array(sparse['X'].values, dtype = 'uint16')
        Y = np.array(sparse['Y'].values, dtype = 'uint16')
        L = np.array(sparse['Label'].values, dtype = 'str')  

        # By plugging in the annotation locations into the segmented image
        # you get all of the segment IDs that contain annotations (desired segments, DS)
        # Then for each DS, find the class label for the annotations within it (color label, CL)
        DS = segmented_image[Y, X]                               
        CL = np.array([labels.index(L[i]) for i in range(len(DS))])
        
        # If multiple annotations are within the same segment, find
        # the most common label among those annotations and provide it
        # as the final label for that segment (majority rule)
        if(len(DS) != len(np.unique(DS))):

            for segment_ID in np.unique(DS):
                annotations_in_segment = list(np.where(DS == segment_ID)[0])
                labels_of_annotations = [CL[a] for a in annotations_in_segment]
                most_common_label_in = max(set(labels_of_annotations), key = labels_of_annotations.count)
                CL[annotations_in_segment] = most_common_label_in
        
        # Lastly, combine them into a dictionary, which speeds up the the next process by 50%
        pairs = dict(zip(DS, CL))

        # temporary mask that holds the labels for each superpixel
        mask = np.full(shape = image.shape[0:2], fill_value = NO_LABEL)

        for index, segVal in enumerate(list(pairs)):
            mask[segmented_image == segVal] = pairs[list(pairs)[index]]
            # provides each individual pixel with the class label of the superpixel

        # Collects all masks made of this image for all iterations
        all_masks.append(mask)
        
        # helps stave of memory allocation errors
        gc.collect()
                    
    # Now that the loop is over, we have a bunch of segmented images where there are
    # pixels with class labels, and pixels with no label (255). We can combine them
    # following Alonso et al. 2019's join method, or Pierce et al. 2020's mode method
    
    # Starting with the first mask which is the most detailed, we mask it with less a
    # detailed mask that contains more labeled pixels, mask_b. With each iteration, 
    # the class mask (final_mask) fills up with the additional pixels provided by mask_b; 
    # no pixels get replaced by mask_b, they only add to final_mask
    if(method == 'join'):
        
        # First mask is most detailed, start with it
        final_mask = all_masks[0]

        # Then go through the remaining masks
        for _ in all_masks[1:]:
            mask_b = _

            # find the pixels in mask_a that are not labelled, assign them with the labels
            # of those same pixels in mask_b
            final_mask[final_mask == NO_LABEL] = mask_b[final_mask == NO_LABEL] 
            
    # Jordan's method, 'mode'
    else:
        # Returns a 2-d array that matches the size of the original image
        # the mode across the 0-th axis
        final_mask = mode(all_masks)

    
    return final_mask

def decompose_matrix(matrix):

    '''
    Helper function for decomposing the confusion matrix
    '''

    TP = np.diag(matrix);
    FP = np.sum(matrix, axis = 0) - TP 
    FN = np.sum(matrix, axis = 1) - TP 

    TN = []
    for _ in range(len(np.diag(matrix))):
        temp = np.delete(matrix, _, 0)    
        temp = np.delete(temp, _, 1)  
        TN.append(sum(sum(temp)))

    return TN, FP, FN, TP

def get_scores(ground_truth_files, prediction_files, labels):

    '''
    Calculates scores for the masks generated by fast_mss against the ground-truth dense annotations
    Based on the code written by Alonso et al. 2019, with some extra bells and whistles
    Macro and Weighted averages match those output by SciKit-Learn

    Takes in all the ground_truth and prediction files as two seperate lists containing
    the file paths, opens each one and compares the two by calculating a confusion matrix
    From the matrix other metrics are calculated (i.e. globally)

    ground_truth_files   --> a list containing the absolute paths of each ground_truth image file
    prediction_files     --> a list containing the absolute paths of each mask image file
    labels               --> a list of all possible labes in the dataset

    NOTE this function assumes that *_files are of equal length, contains files for images,
    and are sorted so the ground_truth and prediction files correspond
    '''
    # checks that lists are equal length and contains image files
    assert len(ground_truth_files) == len(prediction_files)
    assert ground_truth_files[0].split(".")[-1].lower() in ['png', 'jpg', 'jpeg', 'bmp']
    assert prediction_files[0].split(".")[-1].lower() in ['png', 'jpg', 'jpeg', 'bmp']

    num_files = len(ground_truth_files)
    num_classes = len(labels)  

    # For calculating the overall accuracy 
    flat_prediction = []
    flat_ground_truth = []

    # Counters variables
    correct = np.zeros(num_classes)
    actual = np.zeros(num_classes) 
    predicted = np.zeros(num_classes)
    matrix = np.zeros((num_classes, num_classes), np.uint32)
    
    # Loops through each file
    for _ in range(num_files):

        # opens both files
        prediction = io.imread(prediction_files[_])
        ground_truth = io.imread(ground_truth_files[_]) 

        # accumlates a flattened version of each file
        flat_prediction += prediction.flatten().astype(int).tolist()
        flat_ground_truth += ground_truth.flatten().astype(int).tolist()

        # Computes predicted, correct, real and the confusion matrix per file, accumulates all
        for c in range(num_classes):

            # Number of predictions per class
            predicted[c] = predicted[c] + sum(sum((ground_truth >= 0) & (ground_truth < num_classes) & (prediction == c)))

            # Number of correctly predicted samples per class
            correct[c] = correct[c] + sum(sum((ground_truth == c ) & (prediction == c)) )   

            # Number of real samples per class
            actual[c] = actual[c] + sum(sum(ground_truth == c))                   

            # Build a confusion matrix
            for x in range(num_classes):
                matrix[c, x] = matrix[c, x] + sum(sum((ground_truth == c) & (prediction == x)))

    # gets the true positive, false positive, false negative, and true positive   
    TN, FP, FN, TP = decompose_matrix(matrix)

    # normalized matrix
    matrix_normalized = np.around( (matrix / matrix.astype(np.float).sum(axis = 1, keepdims = True)) , 2 )
    
    # Outputs scores -----------------------------------
    print("Relative Abundance: ")
    
    RA= pd.DataFrame(list(zip(labels, np.around(actual/np.sum(actual), 4), np.around(predicted/np.sum(predicted), 4))),
                    columns= ['Class Labels', 'Ground-Truth', 'Predictions'])
    print(RA)
    
    print("\nConfusion Matrix:")
    print(matrix_normalized)

    # Calculates metrics per class (macro), and weights
    p = (TP / (TP + FP))
    r = (TP / (TP + FN))
    dice = (2 * TP) / (TP + FP + FN + TP)
    iou = (TP) / (TP + FP + FN)
    w = actual/np.sum(actual)

    print("\nOverall Accuracy: ", np.sum(np.array(flat_ground_truth) == np.array(flat_prediction))/len(flat_ground_truth) )

    # Per class and Average Class Accuracy
    per_class_accuracy = []
    for _ in range(num_classes):
        per_class_accuracy.append(correct[_]/actual[_])

    print("\nAverage Class Accuracy: ", np.around(np.mean(per_class_accuracy), 4))

    print("\nPer Class Accuracy: ")
    [print(labels[_], ":", round(per_class_accuracy[_], 3)) for _ in range(num_classes)]

    # Macro and weighted metrics
    print("\nMacro: ")
    print("Precision:", np.around(np.mean(p), 4))
    print("Recall:", np.around(np.mean(r), 4))
    print("Dice:", np.around(np.mean(dice), 4))
    print("Iou:", np.around(np.mean(iou), 4))

    print("\nWeighted: ")
    print("Precision:", np.around(np.sum([p[_] * w[_] for _ in range(num_classes)]), 4))
    print("Recall:", np.around(np.sum([r[_] * w[_] for _ in range(num_classes)]), 4))
    print("Dice:", np.around(np.sum([dice[_] * w[_] for _ in range(num_classes)]), 4))
    print("IoU:", np.around(np.sum([iou[_] * w[_] for _ in range(num_classes)]), 4))
