'''
Trains the Tuberculosis net, using a ResNet50 model
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
DATA_PATH = "data/"
MODEL_CHECKPOINT = 'models/resnet50/'
META_NAME = 'resnet50-224x224_train.meta'
CHECKPOINT_NAME = 'resnet50-224x224'

LEARNING_RATE = 0.0001
OUTPUT_PATH = 'models/'
EPOCHS = 81
VALIDATE_EVERY = 5

INPUT_TENSOR = "image:0"
LABEL_TENSOR = "classification/label:0"
LOSS_TENSOR = "add:0"
OUTPUT_TENSOR = "softmax_tensor:0" 

MODEL_SAVE_DIR = 'models/trained/'



dsi = TBNetDSI(DATA_PATH)
train_dataset, train_dataset_size, train_batch_size = dsi.get_train_dataset()
val_dataset, _, _ = dsi.get_validation_dataset()
test_dataset, _, _ = dsi.get_test_dataset()


'''
Runs evaluation on the model using the test dataset
'''
def eval(sess, graph, val_or_test="test"):
    if val_or_test != "test" and val_or_test != "val":
        "Evaluation must either be on 'val' or 'test' set."
        return

    image_tensor = graph.get_tensor_by_name(INPUT_TENSOR)
    label_tensor = graph.get_tensor_by_name(LABEL_TENSOR)
    output_tensor = graph.get_tensor_by_name(OUTPUT_TENSOR)
    loss_tensor = graph.get_tensor_by_name(LOSS_TENSOR)

    y_test = []
    predictions = []
    num_evaled = 0
    total_loss = 0

    if val_or_test == "test":
        iterator = test_dataset.make_initializable_iterator()
    else:
        iterator = val_dataset.make_initializable_iterator()
    datasets = {}
    datasets[val_or_test] = {
        'dataset': test_dataset if val_or_test == "test" else val_dataset,
        'iterator': iterator,
        'gn_op': iterator.get_next(),
    }
    sess.run(datasets[val_or_test]['iterator'].initializer)

    while True:
        try:
            data_dict = sess.run(datasets[val_or_test]['gn_op'])
            images = data_dict['image']
            labels = data_dict['label/one_hot'].argmax(axis=1)

            pred = sess.run(output_tensor, feed_dict={image_tensor: images})
            predictions.append(pred.argmax(axis=1))
            y_test.append(labels)
            num_evaled += len(pred)

            if val_or_test == "val":
                total_loss += sess.run(loss_tensor, feed_dict={image_tensor: images, label_tensor: labels})

        except tf.errors.OutOfRangeError:
            print("\tEvaluated {} images.".format(num_evaled))
            break

    if val_or_test == "val":
        print("Minibatch loss=", "{:.9f}".format(total_loss))

    # Generate confusion matrices and other metrics
    matrix = confusion_matrix(np.array(y_test), np.array(predictions))
    matrix = matrix.astype('float')
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Tuberculosis: {1:.3f}'.format(class_acc[0],class_acc[1]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Tuberculosis {1:.3f}'.format(ppvs[0],ppvs[1]))





with tf.Session() as sess:
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(MODEL_CHECKPOINT, META_NAME))
    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(INPUT_TENSOR)
    label_tensor = graph.get_tensor_by_name(LABEL_TENSOR)
    output_tensor = graph.get_tensor_by_name(OUTPUT_TENSOR)
    loss_tensor = graph.get_tensor_by_name(LOSS_TENSOR)

    # Define loss and optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss_tensor)

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # load weights
    saver.restore(sess, os.path.join(MODEL_CHECKPOINT, CHECKPOINT_NAME))

    # save base model
    saver.save(sess, os.path.join(MODEL_SAVE_DIR) + "tbnet_v1")
    print('Saved baseline checkpoint')
    print('Baseline eval:')

    eval(sess, graph, "test")

    # Training cycle
    print('Training started')
    iterator = train_dataset.make_initializable_iterator()
    datasets = {}
    datasets['train'] = {
        'dataset': train_dataset,
        'iterator': iterator,
        'gn_op': iterator.get_next(),
    }
    sess.run(datasets['train']['iterator'].initializer)
    num_batches = train_dataset_size // train_batch_size
    
    progbar = tf.keras.utils.Progbar(num_batches)
    for epoch in range(EPOCHS):
        for i in range(num_batches):
            # Run optimization
            data_dict = sess.run(datasets['train']['gn_op'])
            batch_x = data_dict['image']
            batch_y = data_dict['label/one_hot'].argmax(axis=1)
            sess.run(train_op, feed_dict={image_tensor: batch_x,
                                            label_tensor: batch_y})
            progbar.update(i+1)

        if epoch % VALIDATE_EVERY == 0:
            eval(sess, graph, "val")
            saver.save(sess, os.path.join(MODEL_SAVE_DIR) + "tbnet_v1", global_step=epoch+1, write_meta_graph=False)
            print('Saving checkpoint at epoch {}'.format(epoch + 1))
    
print("Optimization Finished!")