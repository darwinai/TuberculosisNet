import os
import numpy as np
import glob
import tensorflow as tf
import random

BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 1 # Must be 1
BATCH_SIZE_TEST = 1 # Must be 1
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 2

# Chance that a random augmentation will be performed on an image
AUGMENTATION_CHANCE = 0.5

DATA_PATH = '/gensynth/workspace/data/tb-chest-xray-cropped-v2/'
MASK_IMAGE = '/gensynth/workspace/data/tb-chest-xray-cropped-v2/tb_mask.png'


def parse_function(filename, label):
    img_decoded = tf.image.decode_image(tf.io.read_file(filename), channels=3, expand_animations=False)
    # Crop 5% from each side, but 25% from the bottom
    img = tf.image.crop_to_bounding_box(img_decoded, 11, 11, 168, 202)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)/255.0



    # Convert the label into one hot, return
    return {'image': img,
        'label/one_hot': tf.one_hot(label, NUM_CLASSES),
        'label/value': label,
        'placeholder': tf.convert_to_tensor(1, dtype=tf.float32) }


# Includes random augmentation
def parse_function_train(filename, label):
    img_decoded = tf.image.decode_image(tf.io.read_file(filename), channels=3, expand_animations=False)
    # Crop 5% from each side, but 25% from the bottom
    img = tf.image.crop_to_bounding_box(img_decoded, 11, 11, 168, 202)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)/255.0

    # Chance for augmentation
    if random.random() < AUGMENTATION_CHANCE:
        # Select an augmentation randomly to perform. randint is inclusive.
        which_aug = random.randint(0,3)
        if which_aug == 0:
            img = tf.image.random_crop(img, [202,202,3])
            img = tf.image.resize_images(img, [224,224])
        elif which_aug == 1:
            img = tf.image.random_flip_left_right(img)
        elif which_aug == 2:
            img = tf.image.random_brightness(img, 0.1)
        elif which_aug == 3:
            img = tf.image.random_contrast(img, 0, 0.2)
            
    # Mask out the top left and top right corners
    mask = tf.image.decode_image(tf.io.read_file(MASK_IMAGE), channels=3, expand_animations=False)
    ones_arr = tf.ones(tf.shape(mask))
    ones_arr = tf.dtypes.cast(ones_arr, tf.uint8)
    # This makes a boolean mask, where black areas as False and white is True
    mask = tf.greater(mask, ones_arr)
    # Apply the mask onto the image, thus masking the corners
    img = tf.where(mask, img, tf.zeros_like(img))

    # Convert the label into one hot, return
    return {'image': img, 
        'label/one_hot': tf.one_hot(label, NUM_CLASSES),
        'label/value': label,
        'placeholder': tf.convert_to_tensor(1, dtype=tf.float32) }



class TBNetDSI:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path

    def create_split(self, which_set, num_shards=1, shard_index=0):
        '''
        Reads in the images in the data folder, and randomly splits 
        them into train/val/test sets, ratio of 80/10/10
        '''
        norm_filenames = []
        norm_labels = []
        tb_filenames = []
        tb_labels = []
        for filename in glob.glob1(os.path.join(self.data_path, 'Normal'), '*.png'):
            norm_filenames.append( os.path.join(self.data_path, 'Normal', filename) )
            norm_labels.append(0)
        for filename in glob.glob1(os.path.join(self.data_path, 'Tuberculosis'), '*.png'):
            tb_filenames.append( os.path.join(self.data_path, 'Tuberculosis', filename) )
            tb_labels.append(1)

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

        num_data = num_data_normal + num_data_tb

        if (which_set == "train"):
            dataset = tf.data.Dataset.from_tensor_slices((np.array(train_x), np.array(train_y)))
            dataset = dataset.repeat()
            dataset = dataset.shuffle(5000)
            dataset = dataset.map(map_func=parse_function_train)
            batch_size = BATCH_SIZE_TRAIN
        elif (which_set == "val"):
            dataset = tf.data.Dataset.from_tensor_slices((np.array(val_x), np.array(val_y)))
            #dataset = dataset.repeat()
            dataset = dataset.map(map_func=parse_function)
            batch_size = BATCH_SIZE_VAL
        else:
            dataset = tf.data.Dataset.from_tensor_slices((np.array(test_x), np.array(test_y)))
            #dataset = dataset.repeat()
            dataset = dataset.map(map_func=parse_function)
            batch_size = BATCH_SIZE_TEST

        dataset = dataset.batch(batch_size=batch_size)
        if num_shards > 1:
            dataset = dataset.shard(num_shards, shard_index)
        return dataset, num_data // num_shards, batch_size

    def get_train_dataset(self, num_shards=1, shard_index=0):
        dataset, num_data, batch_size = self.create_split("train", num_shards, shard_index)
        return dataset, num_data, batch_size

    def get_validation_dataset(self, num_shards=1, shard_index=0):
        dataset, num_data, batch_size = self.create_split("val", num_shards, shard_index)
        return dataset, num_data, batch_size

    def get_test_dataset(self):
        dataset, num_data, batch_size = self.create_split("test")
        return dataset, num_data, batch_size