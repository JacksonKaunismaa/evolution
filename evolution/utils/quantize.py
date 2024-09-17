from typing import Tuple
import torch
from torch import Size, Tensor

TensorWithShape = Tuple[Tensor, Size]
SMALL_SCALE_THRESHOLD = 6.1e-5

class QuantizedData:
    def __init__(self, x: TensorWithShape):
        self.data, self.shape = x

    def _dequantize(self, y: Tensor, shape: Size, map_location: torch.device) -> Tensor:
        return y.dequantize().view(shape).to(map_location)

    def dequantize(self, map_location=torch.device('cuda')) -> Tensor:
        return self._dequantize(self.data, self.shape, map_location)


def _quantize(x: Tensor, map_location) -> TensorWithShape:
    maxs = torch.amax(x, dim=0)
    mins = torch.amin(x, dim=0)
    qmin = -128
    qmax = 127

    scales, zero_points = choose_quantization_params(mins, maxs, qmin, qmax,
                                                        reduce_range=False)
    y = torch.quantize_per_channel(x.view(x.shape[0], -1), scales.flatten(), zero_points.flatten(), dtype=torch.qint8, axis=1)
    y = y.to(map_location)
    return y, x.shape


def quantize(x: Tensor, map_location=torch.device('cpu')) -> QuantizedData:
    qdata = _quantize(x, map_location)
    return QuantizedData(qdata)


def choose_quantization_params(mins: Tensor, maxs: Tensor, qmin: int, qmax: int, reduce_range=False):
    """
    Function to compute quantization parameters.
    Adapted from https://github.com/pytorch/pytorch/blob/94f92fbd883605e6f8109e8202a7e9614bcf55a0/aten/src/ATen/native/quantized/cpu/QuantUtils.h#L70

    Parameters:
    mins (torch.Tensor): Tensor of minimum values.
    maxs (torch.Tensor): Tensor of maximum values.
    qmin (int): Minimum quantized value.
    qmax (int): Maximum quantized value.
    reduce_range (bool): Whether to reduce the quantization range.

    Returns:
    scale (torch.Tensor): Tensor of scales for quantization.
    zero_point (torch.Tensor): Tensor of zero points for quantization.
    """

    # Ensure mins and maxs are compatible
    # assert torch.all(mins <= maxs), "min should be less than or equal to max"

    # Adjust qmin and qmax if reducing range
    if reduce_range:
        qmin = qmin // 2
        qmax = qmax // 2

    # Extend [min, max] to contain 0
    mins = torch.minimum(mins, torch.tensor(0.0, device=mins.device))
    maxs = torch.maximum(maxs, torch.tensor(0.0, device=maxs.device))

    # assert qmin < qmax, "qmin should be less than qmax"

    # Compute scale
    scales = (maxs - mins) / (qmax - qmin)

    # Handle small scale or zero scale cases
    if torch.any(scales == 0) or torch.any(torch.isinf(1.0 / scales)):
        scales = torch.where(scales == 0, torch.tensor(0.1, device=scales.device), scales)

    assert torch.all(scales > 0), "quantization scale should be > 0"

    scales = torch.where(scales < SMALL_SCALE_THRESHOLD, SMALL_SCALE_THRESHOLD, scales)

    # Adjust mins and maxs if scales are adjusted
    adjust_condition = scales < SMALL_SCALE_THRESHOLD
    scales = torch.where(adjust_condition, SMALL_SCALE_THRESHOLD, scales)

    mins = torch.where(adjust_condition & (mins == 0), -scales * (qmax - qmin), mins)
    maxs = torch.where(adjust_condition & (maxs == 0), scales * (qmax - qmin), maxs)

    # Compute zero points
    zero_point_from_min = qmin - mins / scales
    zero_point_from_max = qmax - maxs / scales
    min_error = abs(qmin) - torch.abs(mins / scales)
    max_error = abs(qmax) - torch.abs(maxs / scales)

    initial_zero_point = torch.where(min_error < max_error, zero_point_from_min, zero_point_from_max)
    # Force zero_point to be in the middle if preserving sparsity

    # Nudge zero point to be an integer within [qmin, qmax]
    nudged_zero_point = torch.clamp(initial_zero_point, qmin, qmax)

    return scales, nudged_zero_point
