'''
Takes in a folder of chest images and outputs a prediction for each image.
The output will be in a CSV file named "output.csv", located in 
the same folder that the images are contained.

Example command:
python3 inference.py \
    --weightspath 'TB-Net' \
    --metaname model_eval.meta \
    --ckptname model \
    --inputpath 'example_inputs/'
'''

import os
import csv
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import glob

from preprocessing import preprocess_image_inference

INPUT_TENSOR = "image:0"
LOGITS_TENSOR = "resnet_model/final_dense:0" 
INPUT_SIZE = (224,224)

tf.disable_eager_execution()
# Suppress TensorFlow's warning messages
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# mapping
mapping = {0: "Normal", 1: "Tuberculosis"} 

parser = argparse.ArgumentParser(description='TB-Net Inference')
parser.add_argument('--weightspath', default='TB-Net', type=str, help='Path to checkpoint folder')
parser.add_argument('--metaname', default='model_eval.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpt')
parser.add_argument('--inputpath', default='example_inputs/', type=str, help='Full path to folder containing images')

args = parser.parse_args()

# Create a session, load the model
sess = tf.Session()
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

graph = tf.get_default_graph()

# Grab the relevent tensors for inference
image_tensor = graph.get_tensor_by_name(INPUT_TENSOR)
logits_tensor = graph.get_tensor_by_name(LOGITS_TENSOR)

# Grab all of the images in the input folder
image_types = ('*.jpg', '*.png')
files_to_eval = []
for files in image_types:
    files_to_eval.extend(glob.glob(os.path.join(args.inputpath, files)))

# Open output csv file
with open(os.path.join(args.inputpath, 'output.csv'), 'w') as outcsv:
    csvwriter = csv.writer(outcsv, delimiter=',')
    csvwriter.writerow(["Filename", "Prediction", "Confidence"])
    for file in files_to_eval:
        # For each image in the folder, preprocess it in the same way as training
        image = preprocess_image_inference(file)

        logits = sess.run(logits_tensor, feed_dict={image_tensor: [image]})[0]
        softmax = sess.run(tf.nn.softmax(logits))
        pred_class = softmax.argmax()
        confidence = softmax[pred_class]
        csvwriter.writerow([file, mapping[pred_class], confidence])

print("Results generated!")
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')
