import re
import glob
import os.path as osp
from cuda import cuda, nvrtc
from typing import Dict, List
import torch
import numpy as np

from .config import Config, FunctionExpression


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
    

class CUDAKernelManager:
    def __init__(self, config: Config):
        self.config = config
        self.encoding = 'ascii'
        checkCudaErrors(cuda.cuInit(0))
        self.cuDevice = checkCudaErrors(cuda.cuDeviceGet(0))

        major = checkCudaErrors(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 
                                                          self.cuDevice))
        minor = checkCudaErrors(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 
                                                          self.cuDevice))
        arch_arg = f'--gpu-architecture=compute_{major}{minor}'.encode(self.encoding)
        self.compile_args = [b'--use_fast_math', b'--extra-device-vectorization', b'--device-debug', b'--generate-line-info', 
                             arch_arg]
        self.kernels = self.compile_kernels()
        self.stream = checkCudaErrors(cuda.cuStreamCreate(0))

        
    def get_macros(self, code: str) -> List[bytes]:
        # Pattern to match CFG_ symbols
        variable_pattern = re.compile(r'CFG_(\w+)')
        
        # find matching entries in cfg
        macros = set()
        for match in variable_pattern.finditer(code):
            variable = match.group(1)
            value = getattr(self.config, variable, None)
            if value is None:
                raise ValueError(f'Unknown attribute of config: {variable}')
            elif isinstance(value, FunctionExpression):
                macros.add(f'--define-macro={match.group(0)}({", ".join(value.symbols)})={value.expr}'.encode(self.encoding))
            else:
                macros.add(f'--define-macro={match.group(0)}={value}'.encode(self.encoding))
        return list(macros)


    def compile_kernels(self) -> Dict[str, cuda.CUfunction]:
        # Get all .cu files in the directory
        cu_files = glob.glob(osp.join('./kernels', '*.cu'))
        kernels = {}
        # Compile all the .cu files
        for cu_file in cu_files:
            print("Compiling", cu_file)
            # Read the contents of the file
            with open(cu_file, 'r') as f:
                code = f.read()

            name,ext = osp.splitext(osp.basename(cu_file))  # take basename and strip off the extension
            
            # Preprocess the code
            macros = self.get_macros(code)
            print("macros are", macros)
            args = self.compile_args + macros
            nvrtc_prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(code.encode(self.encoding), 
                                                                  cu_file.encode(self.encoding), 0, [], []))
            checkCudaErrors(nvrtc.nvrtcCompileProgram(nvrtc_prog, len(args), args))
            ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(nvrtc_prog))
            ptx = b" " * ptxSize
            print(ptxSize)
            checkCudaErrors(nvrtc.nvrtcGetPTX(nvrtc_prog, ptx))
            ptx = np.char.array(ptx)

            module = checkCudaErrors(cuda.cuModuleLoadData(ptx.ctypes.data))
            cuda_func = checkCudaErrors(cuda.cuModuleGetFunction(module, name.encode(self.encoding)))
            kernels[name] = cuda_func
        return kernels
    
    def __call__(self, func_name: str, blocks_per_grid, threads_per_block, *args):
        func = self.kernels[func_name]
        # extend blocks_per_grid and threads_per_block to ensure they are 3D
        if isinstance(blocks_per_grid, int):
            blocks_per_grid = (blocks_per_grid,)
        if isinstance(threads_per_block, int):
            threads_per_block = (threads_per_block,)
        blocks_per_grid = blocks_per_grid + (3 - len(blocks_per_grid)) * (1,)
        threads_per_block = threads_per_block + (3 - len(threads_per_block)) * (1,)


        # first = args[0]
        # second = args[1]
        # args = (second, first, *args[2:])
        # dummy = torch.rand(5, device='cuda')
        # args = (dummy, *args)
        cuda_args = []
        for arg in args:
            data = arg
            if isinstance(arg, torch.Tensor):
                print(arg.dtype, arg.device)
                data = [int(arg.contiguous().data_ptr())]
            cuda_args.append(np.array(data, dtype=np.uint64))
            print(arg, '->', cuda_args[-1])
        print('BEFORE', func_name, cuda_args)
        cuda_args = np.array([a.ctypes.data for a in cuda_args], dtype=np.uint64)
        print('AFTER', func_name, cuda_args)
        print(blocks_per_grid, threads_per_block)
        checkCudaErrors(cuda.cuLaunchKernel(func, 
                                            *blocks_per_grid,
                                            *threads_per_block,
                                            0,   # dynamic shared mem
                                            self.stream,   # stream
                                            cuda_args,  # kernel args
                                            0))   # kernel params



