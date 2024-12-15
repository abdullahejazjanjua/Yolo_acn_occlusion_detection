import torch  
import torch.nn as nn  
import torch.optim as optim 
import torch.nn.functional as F  
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms 
import einops
from collections import namedtuple