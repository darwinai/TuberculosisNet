'''
Preprocesses the dataset into the same format that DarwinAI used.
After running this script, the data is compatible
with the given data interface, in 'dsi.py'

Note: The DarwinAI team has removed some of the images in the 
original dataset from the training protocol. These images will 
still be processed in the dataset, but will not be used, and are not
included in the train/val/test split csv files.
'''

import os
import csv
import numpy as np
import argparse
import glob
import cv2

from preprocessing import preprocess_image


parser = argparse.ArgumentParser(description='TB-Net Dataset Creation')
parser.add_argument('--datapath', default='Dataset/', type=str, help='The root folder containing the "Normal" and "Tuberculosis" folders.')
parser.add_argument('--outputpath', default='data/', type=str, help='Output path where the new dataset will be saved.')

args = parser.parse_args()

# Check for output folder, if it doesnt exist then create it
if not os.path.exists(args.outputpath):
    os.mkdir(args.outputpath)

# Grab all the images from both folders.
filenames = []
for filename in glob.glob1(os.path.join(args.datapath, 'Normal'), '*.png'):
    filenames.append( os.path.join(args.datapath, 'Normal', filename) )

for filename in glob.glob1(os.path.join(args.datapath, 'Tuberculosis'), '*.png'):
    filenames.append( os.path.join(args.datapath, 'Tuberculosis', filename) )

# For each image, preprocess it and add it to the new dataset.
count = 0
for filename in filenames:
    image = preprocess_image(filename)

    # Save the image as png
    savepath = os.path.join(args.outputpath, filename.split("/")[-1])[:-3] + "png"
    cv2.imwrite(savepath, image)

    # Print progress count
    if count % 500 == 0: 
        print("{} images remaining.".format(len(filenames) - count))
    count += 1

print("Done!")
