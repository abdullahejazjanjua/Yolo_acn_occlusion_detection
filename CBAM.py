from libs import *

class ECA(nn.Module):
    # N C H W
    def __init__(self, in_cha):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # N C 1 1
        self.max_pool = nn.AdaptiveMaxPool2d(1) # N C 1 1
        # Find how important each channel is. 
        self.conv = nn.Conv1d(in_channels=in_cha, out_channels=in_cha, kernel_size=5, padding='same')

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)

        x = F_avg + F_max # ( (N, C, 1, 1) + (N, C, 1, 1) ) 
        x = einops.rearrange(x, "N C 1 1 -> N C 1")
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x
    

class SAM(nn.Module):
    def __init__(self, in_cha):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv2d = nn.Conv2d(in_channels=in_cha * 2, out_channels=in_cha, padding='same', kernel_size=(7,7))

    def forward(self, x):
        F_avg = self.avg_pool(x) # N C 1 1
        F_max = self.max_pool(x) # N C 1 1
  
        F = torch.cat((F_avg, F_max), dim = 1) # N 2*C 1 1
 
        x = self.conv2d(F) # N C 1 1

        x = torch.sigmoid(x)

        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=stride)
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=out_channels, padding='same')
        self.pointwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.channel_attention = ECA(out_channels)
        self.spatial_attenton = SAM(out_channels)

    
        self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
    def forward(self, x):
        F = self.conv2d(x)

        F = self.depthwise_conv(F)
        F = self.pointwise_conv(F)

        F1 = self.channel_attention(F)
        F1 = einops.rearrange(F1, "N C 1 -> N C 1 1")
        F = F1 * F



        F2 = self.spatial_attenton(F)
        F = F2 * F

        x = self.match_channels(x)
        x = x + F

        return x


if __name__ == "__main__":
    # Testing
    in_channels = 64 
    out_channels = 128  
    x = torch.randn(2, in_channels, 208, 208) 


    model1 = ResBlock(in_channels, out_channels, stride=2)  
    output1 = model1(x)
    print(f"Output 1 shape: {output1.shape}")  
    
    model2 = ResBlock(128, 256, stride=2)   
    output2 = model2(output1)
    print(f"output 2 shape: {output2.shape}")
