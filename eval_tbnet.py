'''
Trains the Tuberculosis net, using a ResNet50 model
'''

import os
import sys
import random
import math
import cv2
import csv
import tensorflow.compat.v1 as tf
import numpy as np
from dsi import *

from preprocessing import preprocess_image
from sklearn.metrics import confusion_matrix

tf.disable_eager_execution()
# Suppress TensorFlow's warning messages
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)


# Parameters
DATA_PATH = "example_input/"

'''
GS_TBNet
[[348.   0.]
 [  1. 346.]]
Sens Normal: 1.000, Tuberculosis: 0.997
PPV Normal: 0.997, Tuberculosis 1.000

'''
MODEL_CHECKPOINT = 'models/gs_tbnet_v1_experiment8-c4'
META_NAME = 'model_eval.meta'
CHECKPOINT_NAME = 'model-72417'

INPUT_TENSOR = "image:0"
LOGITS_TENSOR = "resnet_model/final_dense:0" 

OUTPUT_CSV = 'example_input/output.csv'



with tf.Session() as sess:
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(MODEL_CHECKPOINT, META_NAME))
    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(INPUT_TENSOR)
    logits_tensor = graph.get_tensor_by_name(LOGITS_TENSOR)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # load weights
    saver.restore(sess, os.path.join(MODEL_CHECKPOINT, CHECKPOINT_NAME))

    # Open the input folder
    # For each image in the folder, preprocess it in the same way as training
    image_types = ('*.jpg', '*.png')
    files_to_eval = []
    for files in image_types:
        files_to_eval.extend(glob.glob(os.path.join(DATA_PATH, files)))

    # mapping
    mapping = {0: "Normal", 1: "Tuberculosis"} 

    with open(OUTPUT_CSV, 'w') as outcsv:
        csvwriter = csv.writer(outcsv, delimiter=',')
        csvwriter.writerow(["Filename", "Prediction", "Confidence"])
        for file in files_to_eval:
            image = preprocess_image(file)

            logits = sess.run(logits_tensor, feed_dict={image_tensor: [image]})[0]
            softmax = sess.run(tf.nn.softmax(logits))
            pred_class = softmax.argmax()
            confidence = softmax[pred_class]
            csvwriter.writerow([file, mapping[pred_class], confidence])

print("Results generated!")
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')
