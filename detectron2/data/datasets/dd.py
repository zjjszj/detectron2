import sys
import os
import json
import torch
import math
import torch.nn as nn

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())



def generate_cell_anchors(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    """
    Generate a tensor storing canonical anchor boxes, which are all anchor
    boxes of different sizes and aspect_ratios centered at (0, 0).
    We can later build the set of anchors for a full feature map by
    shifting and tiling these tensors (see `meth:_grid_anchors`).

    Args:
        sizes (tuple[float]):
        aspect_ratios (tuple[float]]):

    Returns:
        Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
            in XYXY format.
    """

    # This is different from the anchor generator defined in the original Faster R-CNN
    # code or Detectron. They yield the same AP, however the old version defines cell
    # anchors in a less natural way with a shift relative to the feature grid and
    # quantization that results in slightly different sizes for different aspect ratios.
    # See also https://github.com/facebookresearch/Detectron/issues/227

    anchors = []
    for size in sizes:
        area = size ** 2.0
        for aspect_ratio in aspect_ratios:
            # s * s = w * h
            # a = h / w
            # ... some algebra ...
            # w = sqrt(s * s / a)
            # h = a * w
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])
    return torch.tensor(anchors)

def _calculate_anchors(sizes, aspect_ratios):
    cell_anchors = [
        generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
    ]
    print(cell_anchors)
    return BufferList(cell_anchors)



def _create_grid_offsets(size, stride: int, offset: float):
    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride, grid_width * stride, step=stride, dtype=torch.float32
    )
    shifts_y = torch.arange(
        offset * stride, grid_height * stride, step=stride, dtype=torch.float32
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y

for x in [2,3]:
    if x==2:
        print('dd')