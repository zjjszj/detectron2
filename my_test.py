import os
import torch
import torch.nn.functional as F
import math

a=torch.randn((2,6,4))
b=torch.randn((2,10,4))
print(torch.cat((a,b),dim=1).shape)