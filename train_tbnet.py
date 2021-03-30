'''
Trains an untrained version of TB-Net

python3 train_tbnet.py \
    --weightspath 'TB-Net' \
    --metaname model_train.meta \
    --ckptname model \
    --datapath 'data/' \
    --epochs 10 
'''

import os
import argparse
import tensorflow.compat.v1 as tf
import numpy as np
from dsi import *

from sklearn.metrics import confusion_matrix

tf.disable_eager_execution()
# Suppress TensorFlow's warning messages
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUT_TENSOR = "image:0"
LABEL_TENSOR = "classification/label:0"
LOSS_TENSOR = "add:0"
PREDICTION_TENSOR = "ArgMax:0" 

parser = argparse.ArgumentParser(description='TB-Net Training')
parser.add_argument('--weightspath', default='TB-Net', type=str, help='Path to checkpoint folder')
parser.add_argument('--metaname', default='model_train.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpt')
parser.add_argument('--datapath', default='data/', type=str, help='Root folder containing the dataset')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--savepath', default='models/', type=str, help='Folder for models to be saved in')

args = parser.parse_args()

LEARNING_RATE = args.lr
OUTPUT_PATH = args.savepath
EPOCHS = args.epochs
VALIDATE_EVERY = 5

'''
Runs evaluation on the model using the test dataset
'''
def eval(sess, graph, val_or_test, dataset, image_tensor, label_tensor, pred_tensor, loss_tensor):
    y_test = []
    predictions = []
    num_evaled = 0
    total_loss = 0

    iterator = dataset.make_initializable_iterator()
    datasets = {}
    datasets[val_or_test] = {
        'dataset': dataset,
        'iterator': iterator,
        'gn_op': iterator.get_next(),
    }
    sess.run(datasets[val_or_test]['iterator'].initializer)

    while True:
        try:
            data_dict = sess.run(datasets[val_or_test]['gn_op'])
            images = data_dict['image']
            labels = data_dict['label/one_hot'].argmax(axis=1)

            pred = sess.run(pred_tensor, feed_dict={image_tensor: images})
            predictions.append(pred)
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
    print(matrix)
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Tuberculosis: {1:.3f}'.format(class_acc[0],class_acc[1]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Tuberculosis {1:.3f}'.format(ppvs[0],ppvs[1]))


# Load the datasets
dsi = TBNetDSI(data_path=args.datapath)
train_dataset, train_dataset_size, train_batch_size = dsi.get_train_dataset()
val_dataset, _, _ = dsi.get_validation_dataset()
test_dataset, _, _ = dsi.get_test_dataset()

sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))

graph = tf.get_default_graph()

image_tensor = graph.get_tensor_by_name(INPUT_TENSOR)
label_tensor = graph.get_tensor_by_name(LABEL_TENSOR)
pred_tensor = graph.get_tensor_by_name(PREDICTION_TENSOR)
loss_tensor = graph.get_tensor_by_name(LOSS_TENSOR)

# Define loss and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_tensor)

# Initialize the variables
init = tf.global_variables_initializer()
sess.run(init)

# Load weights
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

# save base model
save_path = os.path.join(OUTPUT_PATH, "Baseline/TB-Net")
os.makedirs(save_path, exist_ok=True)
saver.save(sess, save_path)
print('Saved baseline checkpoint to {}.'.format(save_path))
print('Baseline eval:')

eval(sess, graph, "test", test_dataset, image_tensor, label_tensor, pred_tensor, loss_tensor)

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
for epoch in range(args.epochs):
    for i in range(num_batches):
        # Run optimization
        data_dict = sess.run(datasets['train']['gn_op'])
        batch_x = data_dict['image']
        batch_y = data_dict['label/one_hot'].argmax(axis=1)
        sess.run(train_op, feed_dict={image_tensor: batch_x,
                                        label_tensor: batch_y})
        progbar.update(i+1)

    if epoch % VALIDATE_EVERY == 0:
        eval(sess, graph, "val", val_dataset, image_tensor, label_tensor, pred_tensor, loss_tensor)
        saver.save(sess, os.path.join(OUTPUT_PATH, "Epoch_" + str(epoch), "TB-Net"), global_step=epoch+1, write_meta_graph=False)
        print('Saving checkpoint at epoch {}'.format(epoch + 1))

print("Optimization Finished!")