import os
import torch
import torch.nn.functional as F
import math


@torch.jit.script
class Aa(object):
    def __init__(self, x):
        self.x=x
    def aug_x(self):
        self.x+=1

a=Aa(2)
a.aug_x()
print(a.x)