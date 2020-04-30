import cv2
import numpy as np
from skimage.io import imread
from glog import logger


def read_img(x: str):
    img = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
    if img is None:
        logger.warning(f"Can not read image {x} with OpenCV, switching to scikit-image")
        img = imread(x)
    return img


def read_mask(x: str):
    mask = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning(f"Can not read image {x} with OpenCV, switching to scikit-image")
        mask = imread(x, as_gray=True)
    return np.expand_dims(mask, axis=-1)