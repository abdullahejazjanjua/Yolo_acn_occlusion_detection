from libs import *
from model.head import *
from model.neck import *
from model.backbone import *

class YOLO_ACN(nn.Module):
    def __init__(self, classes, anchors):
        super().__init__()
        self.backbone = BackBone()
        self.neck = neck()
        self.head = head(classes, anchors)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x

    # TO DO: 
        # Process the outputs to make predictions

if __name__ == "__main__":
    x = torch.randn((2, 3, 416, 416))
    model = YOLO_ACN(80, 3)
    x = model(x)
    print(f"x3 shape: {x[0].shape}")
    print(f"x4 shape: {x[1].shape}")
    print(f"Output shape: {x[2].shape}")