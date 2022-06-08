from volksdep.converters import load
import pdb
import time
import argparse
from torchvision import transforms
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image
import tensorrt as trt

trt.init_libnvinfer_plugins(None, "")
trt_model = load('test.trt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "TensorRT do inference")
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size')
    parser.add_argument("--img_path", type=str, default='test_input/1.png', help='cache_file')
    parser.add_argument("--engine_file_path", type=str, default='testv2.trt', help='engine_file_path')
    args = parser.parse_args()

    img = Image.open(args.img_path)
    input_shape = (1, 3, 480, 854)
    transform = transforms.Compose([
        transforms.Resize([input_shape[2], input_shape[3]]),  # [h,w]
        transforms.ToTensor()
        ])
    img = transform(img).unsqueeze(0).cuda()
    # pdb.set_trace()
    # img = img.numpy()
    with torch.no_grad():
        t1 = time.time()
        trt_output = trt_model(img)
        print(time.time()-t1)
        # trt_output = trt_output.squeeze(0)
        # trt_output = trt_output.permute(1, 2, 0)
        # pdb.set_trace()
        # _output = trt_output.detach().numpy()
        # img = transforms.ToPILImage(image) 
        trt_output = trt_output*255
        save_image(trt_output, 'tt.png')
        
        



