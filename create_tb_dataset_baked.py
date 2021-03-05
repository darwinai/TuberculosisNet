'''
Given the original kaggle dataset, create the TB dataset we use for GS.
Clean up the TB dataset by removing or editing images that have too much padding
'''

import os
import sys
import random
import math
import cv2
import glob
import numpy as np
import csv

# Parameters
DATA_PATH = "data/"
RESIZE_SHAPE = (224, 224)
SAVE_PATH = "tbnet_dataset_baked/"
SAVE_IMAGE_PATH = "data/"


# Load in the list of images we want to exclude.]
exclude_files = []
with open('exclude_images.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csvreader:
        exclude_files.append(row[0])

# Check for output folder, if it doesnt exist then create it
check_dir = os.path.join(SAVE_PATH, "data")
if not os.path.exists(check_dir):
    os.mkdir(check_dir)

# Grab all of the images, excluding the ones we dont want to keep
norm_filenames = []
norm_labels = []
tb_filenames = []
tb_labels = []
for filename in glob.glob1(os.path.join(DATA_PATH, 'Normal'), '*.jpg'):
    if filename not in exclude_files:
        norm_filenames.append( os.path.join(DATA_PATH, 'Normal', filename) )
        norm_labels.append(0)
for filename in glob.glob1(os.path.join(DATA_PATH, 'Tuberculosis'), '*.png'):
    if filename not in exclude_files:
        tb_filenames.append( os.path.join(DATA_PATH, 'Tuberculosis', filename) )
        tb_labels.append(1)

# Split the images into train/val/test splits
# Random shuffle of the data
merged = list(zip(norm_filenames, norm_labels))
random.shuffle(merged)
norm_filenames, norm_labels = zip(*merged)
merged = list(zip(tb_filenames, tb_labels))
random.shuffle(merged)
tb_filenames, tb_labels = zip(*merged)

# Split the data into train/val/test
num_data_normal = len(norm_filenames)
num_data_tb = len(tb_filenames)
norm_train_end = int(num_data_normal * 0.8)
norm_val_end = int(num_data_normal * 0.9)
tb_train_end = int(num_data_tb * 0.8)
tb_val_end = int(num_data_tb * 0.9)
train_x = np.array(norm_filenames[:norm_train_end] + tb_filenames[:tb_train_end])
train_y = np.array(norm_labels[:norm_train_end] + tb_labels[:tb_train_end])
val_x = np.array(norm_filenames[norm_train_end:norm_val_end] + tb_filenames[tb_train_end:tb_val_end])
val_y = np.array(norm_labels[norm_train_end:norm_val_end] + tb_labels[tb_train_end:tb_val_end])
test_x = np.array(norm_filenames[norm_val_end:] + tb_filenames[tb_val_end:])
test_y = np.array(norm_labels[norm_val_end:] + tb_labels[tb_val_end:])

# Shuffle the merged arrays
merged_train = list(zip(train_x, train_y))
merged_val = list(zip(val_x, val_y))
merged_test = list(zip(test_x, test_y))
random.shuffle(merged_train)
random.shuffle(merged_val)
random.shuffle(merged_test)
train_x, train_y = zip(*merged_train)
val_x, val_y = zip(*merged_val)
test_x, test_y = zip(*merged_test)

# Save to csv
csv_files = ["train_split.csv", "val_split.csv", "test_split.csv"]
data_x = [train_x, val_x, test_x]
data_y = [train_y, val_y, test_y]
for phase in range(len(csv_files)):
    with open(os.path.join(SAVE_PATH, csv_files[phase]), mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for x,y in zip(data_x[phase], data_y[phase]):
            # Modify to reflect the new data path
            new_pth = os.path.join(SAVE_IMAGE_PATH, x.split("/")[-1])
            new_pth = new_pth[:-3] + "png" 
            csv_writer.writerow([new_pth, y])

print("{} samples in train set.".format(len(train_x)))
print("{} samples in val set.".format(len(val_x)))
print("{} samples in test set.".format(len(test_x)))


#=============================================================================
# Start preprocessing of the images
#=============================================================================

def is_padding(row):
    '''
    Detects if a row or column is padding.
    If the row is comprised of the same value, from 0-5, or 250-255 then it is padding
    '''
    padding = False
    for i in range(6):
        if not np.count_nonzero(np.subtract(row,i)):
            padding = True
    for i in range(250, 256):
        if not np.count_nonzero(np.subtract(row,i)):
            padding = True
    return padding

all_filenames = np.concatenate((norm_filenames, tb_filenames))

# Create black and gray masks
avg_train_intensity = (135, 135, 135)

avg_mask = np.zeros((RESIZE_SHAPE[1], RESIZE_SHAPE[0], 3))
black_mask = np.zeros((RESIZE_SHAPE[1], RESIZE_SHAPE[0], 3)).astype(np.uint8)
for r in range(len(avg_mask)):
    for c in range(len(avg_mask[0])):
        # Hardcoded values for the corners.
        if r < 45 and (c < 45 or c > 178):
            avg_mask[r][c] = avg_train_intensity
            black_mask[r][c] = (0, 0, 0)
        else:
            black_mask[r][c] = (255,255,255)


# For each image, perform preprocessing on it
count = 0
for filename in all_filenames:
    # Read in the image as color, then split and use only the B channel
    image_color = cv2.imread(filename, 1)
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

    # Mask the image with black corner mask
    image = cv2.bitwise_and(image, black_mask)

    # Normalize image, by rescaling the intensities to [0-255]
    min_value = np.min(b)
    max_value = np.max(b)
    image = np.divide(np.subtract(image, min_value), max_value) * 255

    # Mask the corners with mean intensity 
    image = cv2.addWeighted(image, 1, avg_mask, 1, 0)
    image = image.astype(np.uint8)
    
    # Save the image
    output_path = os.path.join(SAVE_PATH, SAVE_IMAGE_PATH, filename.split("/")[-1])
    # Change all images to png format
    output_path = output_path[:-3] + "png"
    cv2.imwrite( output_path, image)

    # Print progress count
    count += 1
    if count % 250 == 0: 
        print("{} images processed.".format(count))

print("Done!")
