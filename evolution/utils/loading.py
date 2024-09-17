import os.path as osp
import glob
from typing import Dict
import moderngl as mgl
import PIL.Image
import numpy as np


def load_shaders(shader_path) -> Dict[str, str]:
    shaders = {}
    for path in glob.glob(osp.join(shader_path, '*')):
        fname = osp.basename(path)
        with open(path, 'r') as f:
            shaders[fname] = f.read()
    return shaders


def load_image_as_texture(ctx: mgl.Context, path: str) -> mgl.Texture:
    img = PIL.Image.open(path)
    np_img = np.array(img)
    texture = ctx.texture(np_img.shape[:2], np_img.shape[2], np_img.tobytes()) # type: ignore
    return texture
