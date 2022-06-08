# --- Imports --- #
# -*- coding: utf-8 -*-
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import argparse
import pdb
# from torchvision.utils import save_image
import cv2
trt.init_libnvinfer_plugins(None, "")

def loadEngine2TensorRT(filepath):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

def do_inference(engine, batch_size, input, output_shape):

    # 创建上下文
    context = engine.create_execution_context()
    output = np.empty(output_shape, dtype=np.float32)
    # pdb.set_trace()
    # 分配内存
    d_input = cuda.mem_alloc(1 * input.size * input.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
    # pdb.set_trace()
    bindings = [int(d_input), int(d_output)]

    # pycuda操作缓冲区
    stream = cuda.Stream()
    # 将输入数据放入device
    cuda.memcpy_htod_async(d_input, input, stream)

    start = time.time()
    # 执行模型
    context.execute_async(batch_size, bindings, stream.handle, None)
    # 将预测结果从从缓冲区取出
    cuda.memcpy_dtoh_async(output, d_output, stream)
    end = time.time()

    # 线程同步
    stream.synchronize()

    print("\nTensorRT {} test:".format(engine_path.split('/')[-1].split('.')[0]))
    print("output:", output)
    print("time cost:", end - start)
    return output

def get_shape(engine):
    for binding in engine:
        if engine.binding_is_input(binding):
            input_shape = engine.get_binding_shape(binding)
        else:
            output_shape = engine.get_binding_shape(binding)
    return input_shape, output_shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "TensorRT do inference")
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size')
    parser.add_argument("--img_path", type=str, default='test_input/1.png', help='cache_file')
    parser.add_argument("--engine_file_path", type=str, default='test.trt', help='engine_file_path')
    args = parser.parse_args()

    engine_path = args.engine_file_path
    engine = loadEngine2TensorRT(engine_path)
    # pdb.set_trace()
    img = Image.open(args.img_path)
    
    # input_shape, output_shape = get_shape(engine)
    input_shape = (1, 3, 480, 854)
    output_shape = (1, 3, 480, 854)
    transform = transforms.Compose([
        transforms.Resize([input_shape[2], input_shape[3]]),  # [h,w]
        transforms.ToTensor()
        ])
    img = transform(img).unsqueeze(0)
    # pdb.set_trace()
    img = img.numpy()
    # pdb.set_trace()
    outimg = do_inference(engine, args.batch_size, img, output_shape)
    outimg = np.transpose(outimg[0],(1,2,0))
    outimg = np.clip(outimg, 0, 1)
    # pdb.set_trace()
    outimg = outimg*255
    outimg = np.array(outimg, dtype=np.uint8)
    # save_image(outimg, 'tt.png')
    cv2.imshow("tt.png",outimg)