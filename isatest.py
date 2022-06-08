import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pdb
import torch.nn as nn
import torch
# def loadEngine2TensorRT(filepath):
#     G_LOGGER = trt.Logger(trt.Logger.WARNING)
#     # 反序列化引擎
#     with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
#         return runtime.deserialize_cuda_engine(f.read())
# pdb.set_trace()
# eng = loadEngine2TensorRT('test.trt')
# if eng:
#     print('successful')
# else:
#     print('fail')
input = torch.randn(1, 16, 30, 30)
m = nn.InstanceNorm2d(num_features=16)
output = m(input)
pdb.set_trace()
print(output)


