# --- Imports --- #
# -*- coding: utf-8 -*-

import os
import pdb
from re import S
# from turtle import clear
import torch
import argparse
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from testda import TestData
from model import GridDehazeNet
import time
import cv2
from PIL import Image
from torchvision import transforms
import pdb
import numpy as np
import torch.nn.utils.prune as prune

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-test_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('--result_dir', default='/home/jetson/dingy/video_output/half_uw.mp4')
parser.add_argument('--test_video_dir', default='uw.mp4')
args = parser.parse_args()

network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
test_batch_size = args.test_batch_size
result_dir = args.result_dir
video_data_dir = args.test_video_dir




# --- Gpu device --- #
# pdb.set_trace()
# device_ids = [Id for Id in range(torch.cuda.device_count())]
# pdb.set_trace()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './checkpoint/model'

transform_raw = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# --- Define the network --- #
net = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)


# --- Multi-GPU --- #
net = nn.DataParallel(net)



# --- Load the network weight --- #
net.load_state_dict(torch.load(model_path, map_location=device))
unloader = transforms.ToPILImage()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
# --- Use the evaluation model in testing --- #
net.eval()

#use fp16
net.half()

print('--- Testing starts! ---')

#---------save image---------#
'''
total_time = 0
with torch.no_grad():
    for i,(img,filename) in enumerate(test_data_loader):
        img.to(device)
        t1 = time.time()
        clear = net(img)
        t2 = time.time()
        total_time += t2-t1
        print('single img process time:%.2fs'%(t2-t1))
        output_name_a = os.path.join(result_dir,filename[0] +'.png')
        save_image(clear,output_name_a)

print('total process time:%.2fs'%(total_time))   

'''
capture = cv2.VideoCapture(video_data_dir)
if result_dir!="":
    fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out     = cv2.VideoWriter(result_dir, fourcc, 24, size)       
fps = 0.0
while(True):
    
    # 读取某一帧
    ref, frame = capture.read()
    if not ref:
        break
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    frame = transform_raw(frame)
    frame = frame.unsqueeze(0)
    frame.to(device)

    #use fp16
    frame=frame.type(torch.HalfTensor)

    # pdb.set_trace()
    # 进行检测
    t1 = time.time()
    frame = net(frame)
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    # RGBtoBGR满足opencv显示格式
    frame = frame.cpu().clone()
    frame = frame.squeeze(0)
    frame = frame.permute(1, 2, 0)
    frame = frame.detach().numpy()
    frame = np.clip(frame, 0, 1)
    # pdb.set_trace()
    frame = frame*255
    frame = np.array(frame, dtype=np.uint8)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    # pdb.set_trace()
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("video",frame)
    
    out.write(frame)
    
    c= cv2.waitKey(1) & 0xff 
       
    if c==27:
        capture.release()
        break

print("Video Detection Done!")
capture.release()
# if result_dir!="":
#     print("Save processed video to the path :" + result_dir)
out.release()
cv2.destroyAllWindows()
