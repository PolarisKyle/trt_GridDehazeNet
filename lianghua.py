from model import GridDehazeNet
import torch
import torch.nn as nn
from collections import OrderedDict
import pdb
model_path = './checkpoint/model'

net = GridDehazeNet(height=3, width=6, num_dense_layer=4, growth_rate=16)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net =  nn.DataParallel(net)
old_dict = torch.load(model_path, map_location=device)
new_state_dict = OrderedDict()
for k, v in old_dict.items():
    # pdb.set_trace()
    name = k[7:]
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)
model_fp32 = net.eval()

# q_backend = "qnnpack"  # qnnpack  or fbgemm
# torch.backends.quantized.engine = q_backend
# qconfig = torch.quantization.get_default_qconfig(q_backend)   

# model_fp32.qconfig = qconfig


input_fp32 = torch.randn(1, 3, 480, 640)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Conv2d, torch.nn.ReLU, torch.nn.InstanceNorm2d},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

res = model_int8(input_fp32)
pdb.set_trace()
print(res)