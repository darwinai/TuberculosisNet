'''
Tests a model using a model trained using GenSynth
This script tests the model using the built test set, generated
by first running clean_data on the TB dataset.

If you wish to run evaluation on specific images, use the
eval_tbnet.py script instead.
'''

import os
import sys
import random
import math
import cv2
import tensorflow.compat.v1 as tf
import numpy as np
from dsi import *

from sklearn.metrics import confusion_matrix

tf.disable_eager_execution()
# Suppress TensorFlow's warning messages
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)


# Parameters
DATA_PATH = "data_processed/"

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
PREDICTION_TENSOR = "ArgMax:0" 

dsi = TBNetDSI(data_path=DATA_PATH)
test_dataset, _, _ = dsi.get_test_dataset()

'''
Runs evaluation on the model using the test dataset
'''
def eval(sess, graph):
    image_tensor = graph.get_tensor_by_name(INPUT_TENSOR)
    pred_tensor = graph.get_tensor_by_name(PREDICTION_TENSOR)

    y_test = []
    predictions = []
    num_evaled = 0

    iterator = test_dataset.make_initializable_iterator()

    datasets = {}
    datasets['test'] = {
        'dataset': test_dataset,
        'iterator': iterator,
        'gn_op': iterator.get_next(),
    }
    sess.run(datasets['test']['iterator'].initializer)

    while True:
        try:
            data_dict = sess.run(datasets['test']['gn_op'])
            images = data_dict['image']
            labels = data_dict['label/one_hot'].argmax(axis=1)

            pred = sess.run(pred_tensor, feed_dict={image_tensor: images})
            predictions.append(pred)
            y_test.append(labels)
            num_evaled += len(pred)
        except tf.errors.OutOfRangeError:
            print("\tEvaluated {} images.".format(num_evaled))
            break

    # Generate confusion matrices and other metrics
    matrix = confusion_matrix(np.array(y_test), np.array(predictions))
    matrix = matrix.astype('float')
    print(matrix)
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Tuberculosis: {1:.3f}'.format(class_acc[0],class_acc[1]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Tuberculosis {1:.3f}'.format(ppvs[0],ppvs[1]))



with tf.Session() as sess:
    #tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(MODEL_CHECKPOINT, META_NAME))
    graph = tf.get_default_graph()

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # load weights
    saver.restore(sess, os.path.join(MODEL_CHECKPOINT, CHECKPOINT_NAME))

    print('Testing Model...')

    eval(sess, graph)

print('Testing Complete.')