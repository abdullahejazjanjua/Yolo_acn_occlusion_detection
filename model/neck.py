from model.backbone import *
from libs import *

class neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackBone()
        self.upsample_to_52 = nn.Upsample(size=(52, 52), mode="bilinear", align_corners=False)         
        self.upsample_to_26 = nn.Upsample(size=(26, 26), mode="bilinear", align_corners=False) 

    def forward(self, x):
        x = self.backbone(x)

        x_upsampled_to_52 = self.upsample_to_52(x[1])
        x_52 = torch.cat((x[0], x_upsampled_to_52), dim=1)

        x_upsampled_to_26 = self.upsample_to_26(x[2])
        x_26 = torch.cat((x[1], x_upsampled_to_26), dim=1)

        return (x_52, x_26, x[2])
    

if __name__ == "__main__":
    # Testing
    in_channels = 3
    out_channels = 32
    x = torch.randn((2, in_channels, 416, 416))
    model = neck()
    x = model(x)
    print(f"x3 shape: {x[0].shape}")
    print(f"x4 shape: {x[1].shape}")
    print(f"Output shape: {x[2].shape}")