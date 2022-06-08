
import os
import tensorrt as trt

ONNX_SIM_MODEL_PATH = 'testv3.onnx'
TENSORRT_ENGINE_PATH_PY = 'testv3.trt'

# def build_engine(onnx_file_path, engine_file_path, flop=16):
#     trt_logger = trt.Logger(trt.Logger.VERBOSE)  # trt.Logger.ERROR
#     builder = trt.Builder(trt_logger)
#     network = builder.create_network(
#         1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#     )
    
#     parser = trt.OnnxParser(network, trt_logger)
#     # parse ONNX
#     with open(onnx_file_path, 'rb') as model:
#         if not parser.parse(model.read()):
#             print('ERROR: Failed to parse the ONNX file.')
#             for error in range(parser.num_errors):
#                 print(parser.get_error(error))
#             return None
#     print("Completed parsing ONNX file")
#     builder.max_workspace_size = 1 << 28
#     # default = 1 for fixed batch size
#     builder.max_batch_size = 1
#     # set mixed flop computation for the best performance
#     if builder.platform_has_fast_fp16 and flop == 16:
#         builder.fp16_mode = True

#     if os.path.isfile(engine_file_path):
#         try:
#             os.remove(engine_file_path)
#         except Exception:
#             print("Cannot remove existing file: ",
#                 engine_file_path)

#     print("Creating Tensorrt Engine")

#     config = builder.create_builder_config()
#     # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
#     config.max_workspace_size = 2 << 30
#     config.set_flag(trt.BuilderFlag.FP16)
    
#     last_layer = network.get_layer(network.num_layers-1)
#     network.mark_output(last_layer.get_output(0))
#     engine = builder.build_engine(network, config)
#     with open(engine_file_path, "wb") as f:
#         f.write(engine.serialize())
#     print("Serialized Engine Saved at: ", engine_file_path)
#     return engine

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def onnx2trt(onnx_path, engine_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28 # 256MB
        
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        engine = builder.build_cuda_engine(network)

        with open(engine_path, "wb") as f:
            f.write(engine.serialize())


if __name__ == "__main__":
    onnx2trt(ONNX_SIM_MODEL_PATH, TENSORRT_ENGINE_PATH_PY)
