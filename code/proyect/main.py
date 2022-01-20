import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os
import numpy as np
import tensorrt as trt

def build_engine(onnx_file_path, engine_file_path, flop=16):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)  # trt.Logger.ERROR
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    profile = builder.create_optimization_profile();
    profile.set_shape("imageinput", (1, 12, 1, 5000),(1, 12, 1, 5000),(1, 12, 1, 5000)) 
    parser = trt.OnnxParser(network, trt_logger)
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file")
    builder.max_workspace_size = 1 << 20
    # default = 1 for fixed batch size
    builder.max_batch_size = 1
    # set mixed flop computation for the best performance
    if builder.platform_has_fast_fp16 and flop == 16:
        builder.fp16_mode = True

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot remove existing file: ",
                engine_file_path)

    print("Creating Tensorrt Engine")
    config = builder.create_builder_config()
    #config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS)) #puede que haya que deshabilitar esta opcion si el codigo falla
    config.max_workspace_size = 1 << 20
    config.set_flag(trt.BuilderFlag.FP16)
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)
    print('Done!')
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("Serialized Engine Saved at: ", engine_file_path)
    return engine
    
class ONNXClassifierWrapper():
    def __init__(self, file, num_classes, target_dtype = np.float32):
        
        self.target_dtype = target_dtype
        self.num_classes = num_classes
        self.load(file)
        
        self.stream = None
      
    def load(self, file):
        f = open(file, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        
        
    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype = self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16
        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()
        
    def predict(self, batch): # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)
            
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()
        return self.output
        
def convert_onnx_to_engine(onnx_filename, engine_filename = None, max_batch_size = 32, max_workspace_size = 1 << 30, fp16_mode = True):
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, builder.create_network() as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = max_workspace_size
        builder.fp16_mode = fp16_mode
        builder.max_batch_size = max_batch_size
        print("Parsing ONNX file.")
        with open(onnx_filename, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        print("Building TensorRT engine. This may take a few minutes.")
        engine = builder.build_cuda_engine(network)
        if engine_filename:
            with open(engine_filename, 'wb') as f:
                f.write(engine.serialize())
        return engine, logger


ONNX_SIM_MODEL_PATH = 'prueba/ecg_net.onnx'
TENSORRT_ENGINE_PATH_PY = 'prueba/ecg_net.trt'
if __name__ == "__main__":    
   build_engine(ONNX_SIM_MODEL_PATH, TENSORRT_ENGINE_PATH_PY)
print('Done!')
BATCH_SIZE=6
N_CLASSES=2
PRECISION=np.int32
trt_model = ONNXClassifierWrapper("prueba/ecg_net.trt", [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)
dummy_input_batch = np.ones((BATCH_SIZE, 12, 1, 5000))
predictions = trt_model.predict(dummy_input_batch)
print(predictions)
