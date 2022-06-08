import torch
from volksdep.converters import onnx2trt
from volksdep.calibrators import EntropyCalibrator2
from volksdep.datasets import CustomDataset

model = 'testv2.trt'

## build trt model with fp32 mode
trt_model = onnx2trt(model, fp16_mode=True)


