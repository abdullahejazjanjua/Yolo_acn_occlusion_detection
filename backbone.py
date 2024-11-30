from libs import *
from CBAM import *
from CBH import *

class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cbh = CBH(in_channels=3, out_channels=32)
        self.block1 = ResBlock(in_channels=32 , out_channels=64, stride=2)
        self.block2 = nn.Sequential(
            ResBlock(in_channels=64 , out_channels=128, stride=2),
            ResBlock(in_channels=128 , out_channels=128, stride=1)
        )
        self.block3 = ResBlock(in_channels=128 , out_channels=256, stride=2)
        self.block4 = ResBlock(in_channels=256 , out_channels=512, stride=2)
        # CHECK THIS!!!
        self.block5 = ResBlock(in_channels=512 , out_channels=512, stride=1) 

    def forward(self, x):
        x = self.cbh(x)
        x = self.block1(x)
        x = self.block2(x)
        x3 = x
        x = self.block3(x)
        x4 = x
        x = self.block4(x)
        x = self.block5(x)

        return x, (x3, x4)
    
if __name__ == "__main__":
    # Testing
    in_channels = 3
    out_channels = 32
    x = torch.randn((2, in_channels, 416, 416))
    model = BackBone()
    output, x = model(x)
    print(f"x3 shape: {x[0].shape}")
    print(f"x4 shape: {x[1].shape}")
    print(f"Output shape: {output.shape}")
