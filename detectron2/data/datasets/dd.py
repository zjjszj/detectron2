import sys
import os
import json
import torch
import math
import torch.nn as nn

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

x, y =_create_grid_offsets((23, 37), 32, 0)
shift=torch.stack((x,y,x,y), dim=1)
base_anchors=torch.randn((3,4), dtype=torch.float32)
print((shift.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)).reshape(-1, 4).shape)
anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
