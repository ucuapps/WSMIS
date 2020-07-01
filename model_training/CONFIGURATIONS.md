# Configuration details
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
