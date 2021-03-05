'''
Contains utility functions for preprocessing
Needed for the eval_tbnet and test_tbnet scripts
'''

import os
import sys
import random
import math
import cv2
import tensorflow.compat.v1 as tf
import numpy as np

RESIZE_SHAPE = (224, 224)
MASK_FILE = 'tb_mask.png'


'''
Detects if a row or column is padding.
If the row is comprised of the same value, from 0-5, or 250-255 then it is padding
'''
def is_padding(row):
    padding = False
    for i in range(6):
        if not np.count_nonzero(np.subtract(row,i)):
            padding = True
    for i in range(250, 256):
        if not np.count_nonzero(np.subtract(row,i)):
            padding = True
    return padding

def preprocess_image(image_path):
    '''
    Processes the image according to how it was processed
    first in the clean_data script, then in the GenSynth DSI.

    1. Images had black and white padding removed.
    2. Image cropped to a smaller bounding box.
    3. Image corners masked
    '''

    # Read in the image as color, then split and use only the B channel
    image_color = cv2.imread(image_path, 1)
    b, g, r = cv2.split(image_color)
    height, width = b.shape
    image = b.copy()
    
    # Starting from the top, look at each row and mark those that are entirely black.
    # Stop once we reach a row with a non-black pixel. Repeat starting from the bottom.
    min_y = 0
    for row in image:
        if is_padding(row):
            min_y += 1
        else:
            break
    max_y = height - 1
    for row_index in range(height - 1, 0, -1):
        if is_padding(image[row_index]):
            max_y -= 1
        else:
            break

    # Do the same for columns, left to right and right to left
    min_x = 0
    # To iterate over columns, just iterate over the transpose of the image
    image_transposed = image.T
    for col in image_transposed:
        if is_padding(col):
            min_x += 1
        else:
            break
    max_x = width - 1
    for col_index in range(width - 1, 0, -1):
        if is_padding(image_transposed[col_index]):
            max_x -= 1
        else:
            break

    # Crop the image to these new boundaries
    image = image[min_y:max_y, min_x:max_x]
    # Create a 3 channel image by repeating the blue
    image = cv2.merge([image, image, image])
    # Resize the image
    image = cv2.resize(image, RESIZE_SHAPE, interpolation = cv2.INTER_LANCZOS4)

    # Step 2 - crop to smaller bounding box
    # 
    # in the DSI we crop to the box (11, 11, 168, 202)
    image = image[11:168, 11:202]
    image = cv2.resize(image, RESIZE_SHAPE, interpolation = cv2.INTER_LANCZOS4)

    # Step 3 - apply the mask on top. 
    mask = cv2.imread(MASK_FILE, 0)
    mask = cv2.resize(mask, RESIZE_SHAPE, interpolation = cv2.INTER_LANCZOS4)
    masked = cv2.bitwise_and(image,image,mask = mask)
    masked = masked.astype(np.float32)/255.

    return masked

