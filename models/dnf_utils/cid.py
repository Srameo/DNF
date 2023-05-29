from torch import nn

# CI
class DConv7(nn.Module):
    def __init__(self, f_number, padding_mode='reflect') -> None:
        super().__init__()
        self.dconv = nn.Conv2d(f_number, f_number, kernel_size=7, padding=3, groups=f_number, padding_mode=padding_mode)

    def forward(self, x):
        return self.dconv(x)

# Post-CI
class MLP(nn.Module):
    def __init__(self, f_number, excitation_factor=2) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
        self.pwconv2 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x
    
class CID(nn.Module):
    def __init__(self, f_number, padding_mode) -> None:
        super().__init__()
        self.channel_independent = DConv7(f_number, padding_mode)
        self.channel_dependent = MLP(f_number, excitation_factor=2)

    def forward(self, x):
        return self.channel_dependent(self.channel_independent(x))
