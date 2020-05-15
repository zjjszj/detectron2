import os
import torch
import torch.nn.functional as F
import torch.nn as nn

a=torch.tensor([[6,2],[4,3]])
indexs=[0,0,1,1]
print(a[indexs])