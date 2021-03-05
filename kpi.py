import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


class TB_KPI:
    def __init__(self, folder_name):
        """An object of this class is constructed at the start of each test or
        validation phase.

        Args:
            folder_name - temporary scratch folder on the filesystem that
                may be used to write results. This folder is under the
                configuration output_dir and therefore readable from any
                worker. Each worker, however, is given a unique folder.
        Returns:
            None
        """
        self.y_test = []
        self.pred = []

    def update(self, data, tensor_values):
        """This method is called after each batch of data in test and
        validation phases. Here the labelled data is used as the true values
        for each class, and the predicted values are provided by tensor_values.

        Args:
            data - Data fetched from the IteratorGetNext operation of the
                dataset’s tf.data.Iterator for the given phase, with the same
                (nested) structure as tf.data.dataset.
            tensor_values - A dictionary that maps fetch names (as configured
                by Input Keys) to the values obtained by fetching the
                corresponding tensors. If a tensor is not scalar, the value
                is numpy.array type, having the shape of the corresponding
                tensor.
        Returns:
            None
        """
        preds = tensor_values['predictions']
        labels = data['label/one_hot']

        self.y_test.extend(labels.argmax(axis=1))
        self.pred.extend(preds.argmax(axis=1))

    def get_worker_results(self):
        """Returns the partial result from each worker to be provided in the
        variable 'worker_results_list' to the reduce_all_worker_results
        method.

        Args:
            None
        Returns:
            Any data type that may be pickled for communication between
            workers.
        """
        return {'y_test': self.y_test, 'pred': self.pred}

    def reduce_all_worker_results(self, worker_results_list):
        """Takes the values returned from the get_worker_results methods for
        each worker, combines them, and then calculates metrics from the
        combined data. Returns a dictionary mapping scalar metric values
        to the output keys.

        Args:
            worker_results_list - a list of items returned by each worker's
                get_worker_results() method. This may be an empty list to
                solicit the supported metrics.
        Returns:
            A dictionary mapping metric output keys to scalar values.  The keys
            must match the keys configured when defining the metric. If a
            metric does not have a value (e.g., due to division by zero), it
            should be set to None.
        """
        y_test = []
        pred = []
        for worker in worker_results_list:
            y_test.extend(worker['y_test'])
            pred.extend(worker['pred'])
        y_test = np.array(y_test)
        pred = np.array(pred)

        matrix = confusion_matrix(y_test, pred, labels=[0, 1])
        matrix = matrix.astype('float')
        assert len(matrix) == 2
        class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
        acc = accuracy_score(y_test, pred)
        ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
        f1_scores = f1_score(y_test, pred, average=None, labels=[0, 1])
        return {'class_0_sens': class_acc[0], 'class_1_sens': class_acc[1],
        'class_0_ppv': ppvs[0], 'class_1_ppv': ppvs[1], 'acc': acc, 'f1_class_0': f1_scores[0],
        'f1_class_1': f1_scores[1]}
