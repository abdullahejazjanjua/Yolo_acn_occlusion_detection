from libs import *

class CBH(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.hard_swish = nn.Hardswish()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.hard_swish(x)
        return x
    
if __name__ == "__main__":
    # Testing
    in_channels = 3
    out_channels = 32
    x = torch.randn((2, in_channels, 416, 416))
    model = CBH(in_channels, out_channels)
    output = model(x)
    print(f"Output shape: {output.shape}")
    