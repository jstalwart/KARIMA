import torch
import torch.nn as nn
from KAN.KAN import KAN_layer

class biKAN_layer(nn.Module):
    def __init__(self,
                 in_features:int,
                 out_features: int,
                 p : float = .5,
                 **kwargs
                 ):
        super().__init__()
        self.upper = KAN_layer(in_features, out_features, **kwargs)
        self.drop_up = nn.Dropout(p)
        self.lower = KAN_layer(in_features, out_features, **kwargs)
        self.drop_bot = nn.Dropout(p)


    def from_pretrained(self,
                        upper_path : str,
                        lower_path : str,
                        **kwargs):
        self.upper.load_state_dict(torch.load(upper_path))
        self.lower.load_state_dict(torch.load(lower_path))

    def forward(self, x):
        x1 = self.upper(x)
        x1 = self.drop_up(x1)
        x2 = self.lower(x)
        x2 = self.drop_bot(x2)
        return torch.mul(x1, x2)