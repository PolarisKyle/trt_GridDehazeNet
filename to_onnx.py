import torch
# from volksdep.converters import torch2onnx
from model import GridDehazeNet
from collections import OrderedDict
import pdb

model_path = './checkpoint/model'
net = GridDehazeNet(height=3, width=6, num_dense_layer=4, growth_rate=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
old_dict = torch.load(model_path, map_location=device)
new_state_dict = OrderedDict()
for k, v in old_dict.items():
    # pdb.set_trace()
    name = k[7:]
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)
model_eva = net.eval().cuda()

x = torch.ones(1, 3, 480, 854).cuda()
y = model_eva(x)
# pdb.set_trace()

# torch2onnx(model_eva, x, 'test.onnx', verbose=False)
torch.onnx.export(
        model_eva,
        x,
        'test.onnx',
        input_names=['input.0'],
        output_names=['output.0'],
        opset_version=9,
        do_constant_folding=False,
        verbose=False,
        dynamic_axes=dict())

