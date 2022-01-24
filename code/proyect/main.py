import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import os
import numpy as np
import tensorrt as trt


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def build_engine(engine_file_path):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)  # trt.Logger.ERROR
    with open(engine_file_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, data=None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        if engine.binding_is_input(binding):
            size = int(np.prod(data.shape)) ##PARA LOS MODELOS DE GAITSET, PARA EL RESTO QUITAR
        else:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):

    inputs[0].host = np.ascontiguousarray(inputs[0].host)
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


TENSORRT_ENGINE_PATH_PY = 'model.trt'
if __name__ == "__main__":    
    data = np.ones((1, 12, 1, 5000)).astype(np.float32)
    with build_engine(TENSORRT_ENGINE_PATH_PY) as engine, engine.create_execution_context() as context:
        inputs_trt, outputs_trt, bindings_trt, stream_trt = allocate_buffers(engine, data)
        inputs_trt[0].host = data
        # inputs[0].host = np.abs(inputs[0].host)
        feats = do_inference(context, bindings=bindings_trt, inputs=inputs_trt,
            outputs=outputs_trt, stream=stream_trt)

        print(feats)
