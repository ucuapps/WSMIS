from .base import *
from .classification import *
from .segmentation import *
from .oaa import *
from .crf_segmentation import *


def get_model_adapter(config, log_path):
    if config["task"] == "segmentation":
        return SegmentationModelAdapter(config, log_path)
    elif config["task"] == "cam_generation":
        return ClassificationModelAdapter(config, log_path)
    elif config["task"] == "attention_accumulation":
        return AttentionAccumulationAdapter(config, log_path)
    elif config["task"] == "integral_attention_learning":
        return IALAdapter(config, log_path)
    elif config["task"] == "pneumothorax_classification":
        return PneumothoraxClassificationAdapter(config, log_path)
    elif config["task"] == "crf_segmentation":
        return CRFSegmentationModelAdapter(config, log_path)
    elif config["task"] == "pneumothorax_segmentation":
        return BinarySegmentationAdapter(config, log_path)
    else:
        raise ValueError(f'Unrecognized task [{config["task"]}]')
