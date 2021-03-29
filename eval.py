'''
Evaluates a model on the test split of the dataset.

If you wish to run inference on specific images, use the
inference.py script instead.

Example command:
python3 eval.py \
    --weightspath 'TB-Net' \
    --metaname model_eval.meta \
    --ckptname model \
    --datapath 'data/'
'''

import os
import cv2
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
from dsi import *
from sklearn.metrics import confusion_matrix

INPUT_TENSOR = "image:0"
PREDICTION_TENSOR = "ArgMax:0" 

tf.disable_eager_execution()
# Suppress TensorFlow's warning messages
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='TB-Net Evaluation')
parser.add_argument('--weightspath', default='TB-Net', type=str, help='Path to checkpoint folder')
parser.add_argument('--metaname', default='model_eval.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpt')
parser.add_argument('--datapath', default='data/', type=str, help='Root folder containing the dataset')

args = parser.parse_args()

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


sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))

graph = tf.get_default_graph()

saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

dsi = TBNetDSI(data_path=args.datapath)
test_dataset, _, _ = dsi.get_test_dataset()

print('Testing Model...')
eval(sess, graph)
print('Testing Complete.')
