from model.CBH import *
from libs import *

class head(nn.Module):
    def __init__(self, classes, bounding_boxes=3):
        super().__init__()
        # (4 + 1 + C) × B
        output = (4 + 1 + classes) * bounding_boxes
        self.cbh01 = CBH(384, output, stride=1)
        self.cbh02 = CBH(768, output, stride=1)
        self.cbh03 = CBH(512, output, stride=1)
    
    def forward(self, neck_output):
        refined_52 = self.cbh01(neck_output[0])
        refined_26 = self.cbh02(neck_output[1])
        refined_13 = self.cbh03(neck_output[2])

        return (refined_52, refined_26, refined_13)
    

if __name__ == "__main__":
    neck_output = (torch.randn((2, 384, 52, 52)), torch.randn((2, 768, 26, 26)),\
         torch.randn((2, 512, 13, 13)))
    head = head(3)
    x = head(neck_output)
    print(f"x3 shape: {x[0].shape}")
    print(f"x4 shape: {x[1].shape}")
    print(f"Output shape: {x[2].shape}")
    hll_f = einops.rearrange(x[2], "N (num_boxes P) H W -> N num_boxes H W P", num_boxes = 3)
    print(f"hll_f shape: {hll_f.shape}")
    bbox = hll_f[0][0][1][5][:4]
    objectness_score = hll_f[0][0][1][5][4:5]
    classification_score = hll_f[0][0][1][5][5:] 
    print("For the 0th box:")
    print(f"bbox coordinates: {bbox}")
    print(f"Objectness score: {objectness_score}")
    print(f"Classification score: {classification_score}")

        
        