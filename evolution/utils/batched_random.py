from typing import Dict, List, Union
import torch
from operator import mul
from functools import reduce


from evolution.core.creature_param import CreatureParam

class BatchedRandom:
    def __init__(self, params: List[CreatureParam]):
        self.param_shapes = {}  # shape of the parameter
        self.param_elems = {}    # how many elements of the contiguous random block to take
        for v in params:
            k = v.name
            v_shape = v._shape
            self.param_shapes[k] = v_shape
            if isinstance(v_shape, list):
                self.param_elems[k] = [reduce(mul, s, 1) for s in v_shape]
            else:
                if len(v_shape) == 0:
                    self.param_elems[k] = 1
                else:
                    self.param_elems[k] = reduce(mul, v_shape, 1)
        
        self.gen_size = sum([(sum(v) if isinstance(v, list) else v) for v in self.param_elems.values()])
        self.buffer = None
        self.idx = 0
        
    def generate(self, num):
        self.buffer = torch.randn(num, self.gen_size, device='cuda')
        self.idx = 0
        
    def get(self, param, idx=None):
        if idx is not None:
            acc_size = self.param_elems[param][idx]
            shape = self.param_shapes[param][idx]
        else:
            acc_size = self.param_elems[param]
            shape = self.param_shapes[param]
            
        result = self.buffer[:, self.idx:self.idx+acc_size].view(-1, *shape)
        self.idx += acc_size
        return result