import torch
torch.set_grad_enabled(False)
from cuda import cuda, nvrtc
from contextlib import contextmanager
import moderngl as mgl
from OpenGL.GL import *
from functools import wraps
from collections import defaultdict

BENCHMARK = False
times = defaultdict(float)

def cuda_profile(func):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not BENCHMARK:
            return func(*args, **kwargs)
        start.record()
        res = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times[func.__name__] += start.elapsed_time(end)
        return res
    return wrapper


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        return error
    

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


def memcopy_1d(tensor, ptr):
    checkCudaErrors(cuda.cuMemcpy(ptr, tensor.data_ptr(), tensor.size(0) * tensor.element_size()))

# needs to handle both device and array
def memcopy_2d(tensor, ptr, dst_type='device'):
    cpy = cuda.CUDA_MEMCPY2D()
    cpy.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
    cpy.srcDevice = tensor.data_ptr()
    cpy.srcPitch = tensor.stride(0) * tensor.element_size()
    if dst_type == 'device':
        cpy.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
        cpy.dstDevice = ptr
    elif dst_type == 'array':
        cpy.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_ARRAY
        cpy.dstArray = ptr
    else:
        raise ValueError(f'Invalid dst_type {dst_type}')
    cpy.dstPitch = tensor.stride(0) * tensor.element_size()
    cpy.WidthInBytes = tensor.size(1) * tensor.element_size()
    cpy.Height = tensor.size(0)

    checkCudaErrors(cuda.cuMemcpy2DUnaligned(cpy))

def memcopy_3d(tensor, ptr):
    cpy = cuda.CUDA_MEMCPY3D()
    cpy.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
    cpy.srcDevice = tensor.data_ptr()
    cpy.srcPitch = tensor.stride(0) * tensor.element_size()
    cpy.srcHeight = tensor.size(1)
    cpy.srcDepth = tensor.size(2)
    cpy.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
    cpy.dstDevice = ptr
    cpy.dstPitch = tensor.stride(0) * tensor.element_size()
    cpy.dstHeight = tensor.size(1)
    cpy.dstDepth = tensor.size(2)
    cpy.WidthInBytes = tensor.size(2) * tensor.element_size()
    cpy.Height = tensor.size(1)
    cpy.Depth = tensor.size(0)

    checkCudaErrors(cuda.cuMemcpy3D(cpy))


@contextmanager  # helper func to copy from pytorch into a texture
def activate_texture(img):
    """Context manager simplifying use cuda images"""
    checkCudaErrors(cuda.cuGraphicsMapResources(1, img, None))
    yield checkCudaErrors(cuda.cuGraphicsSubResourceGetMappedArray(img, 0, 0))
    checkCudaErrors(cuda.cuGraphicsUnmapResources(1, img, None))


def copy_to_texture(tensor, cuda_buffer):
    with activate_texture(cuda_buffer) as arr:
        memcopy_2d(tensor, arr, dst_type='array')

@contextmanager
def activate_buffer(cuda_buffer):
    checkCudaErrors(cuda.cuGraphicsMapResources(1, cuda_buffer, None))
    yield checkCudaErrors(cuda.cuGraphicsResourceGetMappedPointer(cuda_buffer))
    checkCudaErrors(cuda.cuGraphicsUnmapResources(1, cuda_buffer, None))


def copy_to_buffer(tensor, cuda_buffer):
    with activate_buffer(cuda_buffer) as (ptr,size):
        if tensor.dim() == 1:
            memcopy_1d(tensor, ptr)
        elif tensor.dim() == 2:
            memcopy_2d(tensor, ptr, dst_type='device')
        elif tensor.dim() == 3:
            memcopy_3d(tensor, ptr)
        else:
            raise ValueError(f'Invalid tensor dim {tensor.dim()}')
        
def register_cuda_buffer(buffer: mgl.Buffer):
    return checkCudaErrors(cuda.cuGraphicsGLRegisterBuffer(
        int(buffer.glo),
        cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_NONE
    ))

def register_cuda_image(image: mgl.Texture):
    return checkCudaErrors(cuda.cuGraphicsGLRegisterImage(
                    int(image.glo),
                    GL_TEXTURE_2D,
                    cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD))