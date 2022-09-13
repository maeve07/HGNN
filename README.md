# HGNN

## Prerequisites

We train our model using PyTorch 1.4.0 with four NVIDIA V100 GPUs with 32GB memory per card.
Other Python modules can be installed by running

```bash
pip install -r requirements.txt
``` 

## Training

### Clone

```git clone https://github.com/maeve07/HGNN```

### Dataset

Please download [PASCAL VOC 2012](https://drive.google.com/file/d/1uh5bWXvLOpE-WZUUtO77uwCB4Qnh6d7X/view) for training, and specify the path of the dataset.

### Classification network
Please run ```python train.py``` for training the classification network.

Then generate the pseudo labels of the training set by  running ```python gen_labels.py```.

### Segmentation network
We use Deeplab-v2 for the segmentation network with our generated pseudo labels. But most popular FCN-like segmentation networks can be used instead.  
