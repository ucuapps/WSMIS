# Weakly Supervised Medical Images Segmentation

This is an implementation of the [Weakly-Supervised Segmentation for DiseaseLocalization in Chest X-Ray Images](https://github.com/ucuapps/WSMIS) by Ostap Viniavskiy, Maria Dobko, and Oles Dobosevych.


## Overview
We proposed a three-step approach to weakly-supervised semantic segmentation, which uses only 
image-level labels as supervision. The method can be applied to various medical problems as well
as on datasets of different nature. We demonstrated the efficiency of this approach for semantic 
segmentation of Pneumothorax on chest X-ray images (SIIM-ACR Pneumothorax). We also evaluated the performance on PASCAL 
VOC 2012 dataset. 


<p align="center"><img src="./images/localization_example.png" alt="outline" width="90%"></p>


## License

WSMIS is released under the [GNU General Public License version 3 license](LICENSE).

## General Pipeline
* Step 1. **Cam extraction.** 
* Step 2. **Boundaries improvements via IRNet.** 
* Step 3. **Segmentation.** 

## Data
#### Download PASCAL VOC 2012 devkit
* Follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
#### Download SIIM-ACR Pneumothorax
* Follow instructions in https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data

## Usage

### Dependencies and Computing resources

All necessary modules are in the requirements.txt
```
pip install -r requirements.txt
```

This code should be compatible with Python versions \>= 3.5.

All the experiments were performed on four Nvidia Tesla K80 GPUs. 

### Training
1. Set parameters for the experiment config.yaml (for detailed explanation of the config file see config section below)
2. Run the following command:
```
python train.py
```
### Testing
For the evaluating of the trained models you should use the jupyter notebooks, which are located
in ...


## Config
Config file contains most of the model hyperparameters along with used model architecture, image preprocessing technique and data augmentation.
Our config is organized as follows:

* `task`: folder name of the experiments, values: `[cam_generation, irn_maps, segmentation]` 
* `log_path`: folder path to the saved logs of the experiment, valus: `[/home/logs]`
* `devices`:  devices names used for training, values: `[[cuda:0], [cuda:0, cuda:1], ...]` 

* `train`:
    * `path`: path to json file for training set, values: `/datasets/LID/PascalVOC2012SegmentationAug/train.json`
    * `transform`:
        * `size`: image input size,  values: `[256, 352]` 
        * `augmentation_scope`: the level of augmentation severity, values: `[none, weak, strong]` 
        * `images_normalization`: normalization for input images, values: `[none, default]` 
        * `images_output_format_type`: type of the image values that are returned after transformations, values: `[float]`  
        * `size_transform`: type of size transformation, values: `[none, resize]`  

* `val`:
    * `path`: path to json file for training set, values: `/datasets/LID/PascalVOC2012SegmentationAug/train.json`
    * `transform`:
        * `size`: image input size,  values need to correspond to the same parameter as in `train`
        * `augmentation_scope`: the level of augmentation severity, values: `[none]` 
        * `images_normalization`: normalization for input images, values values need to correspond to the same parameter as in `train`
        * `images_output_format_type`: type of the image values that are returned after transformations, values values need to correspond to the same parameter as in `train` 
        * `size_transform`: type of size transformation, values: `[none, resize]`  
   
* `model`:
    * `arch`: model's architecture, values: `[deeplabv3plus_resnet50]`
    * `pretrained`: bool value whether to load pretrained weights, values: `[True, False]`
    * `classes`: number of classes, values: `[21, 2]`
    * `loss`:
        * `name`: loss name, values: `[categorical_cross_entropy, binary_cross_entropy]`
        * `ignore_index`: values: `[255]`
    * `metrics`: which metric to calculate during training, values: `[iou]`

* `num_epochs`: number of epochs for training, values: `[50, 100]`
* `batch_size`: how many images to include in one batch, values: `[12, 32, 48]`
* `optimizer`:
    * `name`: optimizer name, values: `[sgd]`
    * `momentum`: values: `[0.9]`
    * `weight_decay`: values: `[0.000001]`
    * `lr`: learning rate, values: `[[0.001, 0.01]]`

* `scheduler`:
    * `name`: scheduler name, values: `[poly]`
    * `max_iters`: the maximum number of iterations, values: `[10, 50, 100]`
    * `min_lr`: the minimum value of learning rate, values: `[1.0e-6]`
    * `power`: values: `[0.9]`

## TO DO
* Add training code for ...
* Code refactoring

## Citation

## References
* Learning Deep Features for Discriminative Localization: [paper](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)
* Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations: [paper](Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations), [code](https://github.com/jiwoon-ahn/irn)
* U-Net: Convolutional Networks for Biomedical Image Segmentation: [paper](https://arxiv.org/abs/1505.04597)
* Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation: [paper](https://arxiv.org/abs/1802.02611)
