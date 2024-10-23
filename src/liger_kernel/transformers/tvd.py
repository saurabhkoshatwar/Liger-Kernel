import torch.nn as nn
from liger_kernel.ops.tvd import LigerTVDLossFunction  

class LigerTVDLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LigerTVDLoss, self).__init__(*args, **kwargs)

    def forward(self, p, q):
        return LigerTVDLossFunction.apply(p, q, self.reduction)
