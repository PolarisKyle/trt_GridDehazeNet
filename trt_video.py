# --- Imports --- #
# -*- coding: utf-8 -*-

import os
import pdb
from re import S
import torch
import argparse
import torch.nn as nn
from volksdep.converters import load
import time
import cv2
from PIL import Image
from torchvision import transforms
import pdb
import numpy as np
import tensorrt as trt
from torchvision import transforms
import numpy as np

trt.init_libnvinfer_plugins(None, "")

trt_model = load('testv3.trt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "TensorRT do inference")
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size')
    parser.add_argument('--result_dir', default='/home/jetson/dingy/video_output/trtv3_uw.mp4')
    parser.add_argument('--test_video_dir', default='uw.mp4')
    args = parser.parse_args()

    transform_raw = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   

    result_dir = args.result_dir
    video_data_dir = args.test_video_dir
    capture = cv2.VideoCapture(video_data_dir)
    if result_dir!="":
        fourcc  = cv2.VideoWriter_fourcc(*'XVID')
        size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out     = cv2.VideoWriter(result_dir, fourcc, 24, size)  
    fps = 0.0
    while(True):
        ref, frame = capture.read()
        if not ref:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform_raw(frame)
        frame = frame.unsqueeze(0)
        frame=frame.type(torch.HalfTensor)
        frame = frame.cuda()
        # pdb.set_trace()
        t1 = time.time()
        output = trt_model(frame)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps))
        _output = output.clone().cpu()
        # pdb.set_trace()
        _output = _output.squeeze(0)
        _output = _output.permute(1, 2, 0)
        _output = _output.detach().numpy()
        _output = np.clip(_output, 0, 1)
        _output = _output*255
        _output = np.array(_output, dtype=np.uint8)
        _output = cv2.cvtColor(_output,cv2.COLOR_RGB2BGR)
        _output = cv2.putText(_output, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video",_output)
        out.write(_output)

        c= cv2.waitKey(1) & 0xff 
       
        if c==27:
            capture.release()
            break

    print("Video Detection Done!")
    capture.release()
    out.release()
    cv2.destroyAllWindows()






"""
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
"""