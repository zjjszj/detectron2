import os
import torch
import torch.nn.functional as F
import math

a=torch.randn((2,6,4))
l=torch.tensor([0,1,0,1,0,1])
b=l==1
print(b)
print(a[b])