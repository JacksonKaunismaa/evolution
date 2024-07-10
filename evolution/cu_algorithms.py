import re
import glob
import os.path as osp
import torch
from cuda import cuda, nvrtc
from typing import Dict, List
import numpy as np

from . import config


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        return error
    # else:
    #     raise RuntimeError('Unknown error type: {}'.format(error))


def checkCudaErrors(result):
    if result[0].value:
        # print(result[0], type(result[0]))
        print(result[0], _cudaGetErrorEnum(result[0]))
        raise RuntimeError(f"CUDA error code={result[0].value}({str(_cudaGetErrorEnum(result[0]))})")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
    

class CUDAKernelManager:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.encoding = 'ascii'
        checkCudaErrors(cuda.cuInit(0))
        self.cuDevice = checkCudaErrors(cuda.cuDeviceGet(0))

        major = checkCudaErrors(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 
                                                          self.cuDevice))
        minor = checkCudaErrors(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 
                                                          self.cuDevice))
        arch_arg = f'--gpu-architecture=compute_{major}{minor}'.encode(self.encoding)
        debug_args =  [b'--device-debug', b'--generate-line-info']
        self.compile_args = [b'--use_fast_math', b'--extra-device-vectorization', arch_arg] #+ debug_args
        self.kernels = self.compile_kernels()
        self.stream = checkCudaErrors(cuda.cuStreamCreate(0))

        
    def get_macros(self, code: str) -> List[bytes]:
        # Pattern to match CFG_ symbols
        variable_pattern = re.compile(r'CFG_(\w+)')
        
        # find matching entries in cfg
        macros = set()
        for match in variable_pattern.finditer(code):
            variable = match.group(1)
            value = getattr(self.cfg, variable, None)
            if value is None:
                raise ValueError(f'Unknown attribute of config: {variable}')
            elif isinstance(value, config.FunctionExpression):
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
            # print("Compiling", cu_file)
            # Read the contents of the file
            with open(cu_file, 'r') as f:
                code = f.read()

            name,ext = osp.splitext(osp.basename(cu_file))  # take basename and strip off the extension

            # Preprocess the code
            macros = self.get_macros(code)
            # print("macros are", macros)
            args = self.compile_args + macros
            nvrtc_prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(code.encode(self.encoding), 
                                                                  cu_file.encode(self.encoding), 0, [], []))
            checkCudaErrors(nvrtc.nvrtcCompileProgram(nvrtc_prog, len(args), args))
            ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(nvrtc_prog))
            ptx = b" " * ptxSize
            # print(ptxSize)
            checkCudaErrors(nvrtc.nvrtcGetPTX(nvrtc_prog, ptx))
            ptx = np.char.array(ptx)

            module = checkCudaErrors(cuda.cuModuleLoadData(ptx.ctypes.data))
            cuda_func = checkCudaErrors(cuda.cuModuleGetFunction(module, name.encode(self.encoding)))

            kernels[name] = cuda_func
        return kernels
    
    class KernelArgs:
        """It's unclear why this class is strictly necessary. I'm assuming it has something to do with 
        how Python's garbage collector works. The problem is that when specifying arguments to the kernel,
        one needs void** pointers, necessitating the use of converting pointers to the data into numpy
        arrays, and then accessing their .ctypes.data attribute:
        
        This works:

        t1 = torch.rand(5, device='cuda')   # tensor we want to pass in to cuda
        t2 = torch.rand(5, device='cuda')
        t1_arg = np.array(t1.data_ptr(), dtype=np.uint64)
        t2_arg = np.array(t2.data_ptr(), dtype=np.uint64)
        args = [t1_arg, t2_arg]
        cuda_args = np.array([a.ctypes.data for a in args], dtype=np.uint64)

        But this doesn't:
        
        args = []
        for arg in [t1, t2]:
            args.append(np.array(arg.data_ptr(), dtype=np.uint64))
        cuda_args = np.array([a.ctypes.data for a in args], dtype=np.uint64)

        Nor does this work:

        args = [np.array(t1.data_ptr(), dtype=np.uint64)]
        args.append(np.array(t2.data_ptr(), dtype=np.uint64))
        cuda_args = np.array([a.ctypes.data for a in args], dtype=np.uint64)

        It's a similar story if args is a dictionary or tuple. The only way I've found that does work 
        without necessitating the creation of a bunch of variables (which doesn't work for variable numbers
        of arguments), is using setattr on some dummy class. That's what this class is for. We define
        it inside CUDAKernelManager because it should only be used here, for this specific purpose.
        """
        def __init__(self, *args):
            self.len = len(args)
            self.var_template = 'a{}'
            for i, value in enumerate(args):
                if isinstance(value, torch.Tensor):
                    value = value.data_ptr()
                setattr(self, self.var_template.format(i), np.array(value, dtype=np.uint64))

        def get_args(self):
            return np.array([getattr(self, self.var_template.format(i)).ctypes.data 
                            for i in range(self.len)], dtype=np.uint64)
    
    def __call__(self, func_name: str, blocks_per_grid, threads_per_block, *args):
        func = self.kernels[func_name]
        # extend blocks_per_grid and threads_per_block to ensure they are 3D
        if isinstance(blocks_per_grid, int):
            blocks_per_grid = (blocks_per_grid,)
        if isinstance(threads_per_block, int):
            threads_per_block = (threads_per_block,)
        blocks_per_grid = blocks_per_grid + (3 - len(blocks_per_grid)) * (1,)
        threads_per_block = threads_per_block + (3 - len(threads_per_block)) * (1,)

        # similar to the weirdness with the KernelArgs class, we need to assign KernelArgs(*args)
        # to a local variable here or it doesn't work
        kernel_args = self.KernelArgs(*args)
        cuda_args = kernel_args.get_args()
        checkCudaErrors(cuda.cuLaunchKernel(func, 
                                            *blocks_per_grid,
                                            *threads_per_block,
                                            0,   # dynamic shared mem
                                            self.stream,   # stream
                                            cuda_args,  # kernel args
                                            0))   # kernel params



