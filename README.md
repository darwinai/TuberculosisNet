# Tuberculosis-Net (TB-Net) #

**Note: The TB-Net models provided here are intended to be used as reference models that can be built upon and enhanced as new data becomes available. They are currently at a research stage and not yet intended as production-ready models (not meant for direct clinical diagnosis), and we are working continuously to improve them as new data becomes available. Please do not use TB-Net for self-diagnosis and seek help from your local health authorities.**

<p align="center">
	<img src="assets/tbnet_rca_tb.png" alt="root cause analysis of a tuberculosis sample image" width="70%" height="70%">
	<br>
	<em>Example chest x-ray images from 2 different patients, and their associated critical factors (highlighted in white) as identified by GSInquire. In this example, both lungs have been diagnosed with tuberculosis.</em>
</p>

Tuberculosis (TB) that remains a global health problem to this very day, and is the leading cause of death from an infectious disease.  A crucial step in the treatment of tuberculosis is screening high risk populations and early detection of the disease, with chest x-ray (CXR) imaging being the most widely-used imaging modality.  As such, there has been significant recent interest in artificial intelligence-based TB screening solutions for use in resource-limited scenarios where there is a lack of trained healthcare workers with expertise in CXR interpretation.  Motivated by this pressing need, we introduce TB-Net, a deep convolutional neural network tailored for TB case screening.  More specifically, machine-driven design exploration was leveraged to build a highly customized deep neural network architecture.  An explainability-driven performance validation process was conducted to validate the decision-making behaviour of TB-Net.  Experiments using the XXX benchmark dataset showed that the proposed TB-Net is able to achieve accuracy/sensitivity/PPV of X%/X%/X%.  We hope that the release of TB-Net will support researchers, clinicians, and citizen data scientists in advancing this field.

If there are any technical questions after the README, FAQ, and past/current issues have been read, please post an issue or contact:
* james.lee@darwinai.ca

## Table of Contents ##
1. [Requirements](#requirements) to install on your system
2. Dataset recreation
3. Steps for [training, evaluation and inference](docs/train_eval_inference.md) of TB-Net
4. [Results](#results)
5. [Links to pretrained models](docs/models.md)

## Requirements ##

The main requirements are listed below. A full list can be found in "requirements.txt"

* Tested with Tensorflow 1.15
* OpenCV 4.5.1
* Python 3.6
* Numpy 1.20.0
* Pandas 1.2.1
* keras 2.2.4

## Dataset Recreation ##

## Results ##



### Files ###

clean_data.py

- After downloading the dataset from the Kaggle link, we preprocess the data initially by removing white/black borders from each image. This script automatically removes those borders, providing a tighter crop for each image.

dsi.py

- The dataset interface used by GenSynth when training/testing the model. Also used by the test_tbnet.py script.

kpi.py

- The performance metric interface used by GenSynth when training/testing the model.

eval_tbnet.py

- Used to run inference on specific images chosen by the user. To use, first place CT scan images into the folder "example_input." Images must be in either JPG or PNG format. Next, run "python3 eval_tbnet.py" and an output csv file will be generated, assuming all dependencies are met.

preprocessing.py

- Contains utility functions for eval_tbnet.

test_tbnet.py

- Calculates certain metrics for the model on the test set, i.e. Sensitivity and PPV.

train_tbnet.py

- Trains the network from scratch using the Tuberculosis dataset. 


### Model ###
This model was training using GenSynth, and achieves the following results on the Tuberculosis Test dataset.

Model is available [here](https://drive.google.com/file/d/1jrkJFq7zsV2extqHpvkxuIrvjGjqU2u7/view?usp=sharing)

gs_tbnet_v1_experiment-c4 model

[[ 348,   0]  
 [   1, 346]]  

Sens Normal: 1.000, Tuberculosis: 0.997  
PPV Normal: 0.997, Tuberculosis 1.000


### How to Reproduce Our Results ###

Dataset from https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset


