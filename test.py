# --- Imports --- #
import os
import pdb
import torch
import argparse
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from testda import TestData
from model import GridDehazeNet
import time
import pdb
 

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-test_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('--result_dir', default='./test_out/')
args = parser.parse_args()

network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
test_batch_size = args.test_batch_size
result_dir = args.result_dir

if not os.path.exists(result_dir):
    os.mkdir(result_dir)



test_data_dir = './test_input/'



# --- Gpu device --- #
# pdb.set_trace()
# device_ids = [Id for Id in range(torch.cuda.device_count())]
# pdb.set_trace()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './checkpoint/model'


# --- Validation data loader --- #
test_data_loader = DataLoader(TestData(test_data_dir), batch_size=test_batch_size, shuffle=False)


# --- Define the network --- #
net = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)


# --- Multi-GPU --- #
net = nn.DataParallel(net)



# --- Load the network weight --- #
net.load_state_dict(torch.load(model_path, map_location=device))


# --- Use the evaluation model in testing --- #
net.eval()
print('--- Testing starts! ---')

#---------save image---------#
total_time = 0
with torch.no_grad():
    for i,(img,filename) in enumerate(test_data_loader):
        img.to(device)
        pdb.set_trace()
        t1 = time.time()
        clear = net(img)
        pdb.set_trace()
        t2 = time.time()
        total_time += t2-t1
        print('single img process time:%.2fs'%(t2-t1))
        output_name_a = os.path.join(result_dir,filename[0] +'.png')
        save_image(clear,output_name_a)

print('total process time:%.2fs'%(total_time))   
         