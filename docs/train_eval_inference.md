# Training, Evaluation and Inference
The TB-Net model takes as input images of shape (N, 224, 224, 3) and outputs the softmax probabilities as (N, 2), where N is the batch size.

If using the TF checkpoints, here are some useful tensors:

* input tensor: `image:0`
* logit tensor: `resnet_model/final_dense:0`
* output tensor: `ArgMax:0`
* loss tensor: `loss/add:0`
* label tensor: `classification/label:0`

If using the provided [dataset interface script](../dsi,py), modify the CSV paths accordingly, to point at where the files 'train_split.csv', 'val_split.csv', and 'test_split.csv' are on your system.

## Training
We provide a [training script](../train_tbnet.py) that can be used for model training using an untrained model. We provide an *untrained* version of TB-Net [here](https://drive.google.com/drive/folders/1z5SI7qTlrd1pjqx0V6HOm_jNZskW15ln?usp=sharing).

Example command:
```
python3 train_tbnet.py \
    --weightspath 'TB-Net' \
    --metaname model_train.meta \
    --ckptname model \
    --datapath 'data/' \
    --epochs 10 
```

## Evaluation
We provide an [evaluation script](../eval.py) that can be used to evaluate a model on the test set. The TB-Net model can be found [here](models.md).

Example command:
```
python3 eval.py \
    --weightspath 'TB-Net' \
    --metaname model_eval.meta \
    --ckptname model \
    --datapath 'data/' \
```

## Inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**

We provide an [inference script](../inference.py) that can be used to evaluate a single image using the given model. The TB-Net model can be found [here](models.md).

Example command:
```
python3 inference.py \
    --weightspath 'TB-Net' \
    --metaname model_eval.meta \
    --ckptname model \
    --inputpath 'example_inputs/'
```
