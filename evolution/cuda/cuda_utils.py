import gc
from contextlib import contextmanager
import torch
from cuda import cuda, nvrtc
import moderngl as mgl
from OpenGL.GL import GL_TEXTURE_2D


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

def _cudaGetErrorEnum(error):  # pylint: disable=invalid-name
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"

    if isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]

    return error


def cudaCheckErrors(result):  # pylint: disable=invalid-name
    if result[0].value:
        # print(result[0], type(result[0]))
        print(result[0], _cudaGetErrorEnum(result[0]))
        raise RuntimeError(f"CUDA error code={result[0].value}({str(_cudaGetErrorEnum(result[0]))})")

    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def memcopy_1d(tensor, ptr):
    cudaCheckErrors(cuda.cuMemcpy(ptr, tensor.data_ptr(), tensor.size(0) * tensor.element_size()))

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

    cudaCheckErrors(cuda.cuMemcpy2DUnaligned(cpy))

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

    cudaCheckErrors(cuda.cuMemcpy3D(cpy))


@contextmanager  # helper func to copy from pytorch into a texture
def activate_texture(img):
    """Context manager simplifying use cuda images"""
    cudaCheckErrors(cuda.cuGraphicsMapResources(1, img, None))
    yield cudaCheckErrors(cuda.cuGraphicsSubResourceGetMappedArray(img, 0, 0))
    cudaCheckErrors(cuda.cuGraphicsUnmapResources(1, img, None))


def copy_to_texture(tensor, cuda_buffer):
    with activate_texture(cuda_buffer) as arr:
        memcopy_2d(tensor, arr, dst_type='array')

@contextmanager
def activate_buffer(cuda_buffer):
    cudaCheckErrors(cuda.cuGraphicsMapResources(1, cuda_buffer, None))
    yield cudaCheckErrors(cuda.cuGraphicsResourceGetMappedPointer(cuda_buffer))
    cudaCheckErrors(cuda.cuGraphicsUnmapResources(1, cuda_buffer, None))


def copy_to_buffer(tensor, cuda_buffer):
    with activate_buffer(cuda_buffer) as (ptr,size):  # type: ignore pylint: disable=unused-variable
        if tensor.dim() == 1:
            memcopy_1d(tensor, ptr)
        elif tensor.dim() == 2:
            memcopy_2d(tensor, ptr, dst_type='device')
        elif tensor.dim() == 3:
            memcopy_3d(tensor, ptr)
        else:
            raise ValueError(f'Invalid tensor dim {tensor.dim()}')

def register_cuda_buffer(buffer: mgl.Buffer):
    return cudaCheckErrors(cuda.cuGraphicsGLRegisterBuffer(
        int(buffer.glo),
        cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_NONE
    ))

def register_cuda_image(image: mgl.Texture):
    return cudaCheckErrors(cuda.cuGraphicsGLRegisterImage(
                    int(image.glo),
                    GL_TEXTURE_2D,
                    cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD))
