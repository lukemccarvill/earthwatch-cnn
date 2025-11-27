import torch.nn as nn # containts layers (convolutions, linear layers, pooling)
import torch.nn.functional as F # contains operations/functions applied to data, like RELU

class Net(nn.Module):
    def __init__(self):
        