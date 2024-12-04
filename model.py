from libs import *
from model.head import *
from model.neck import *
from model.backbone import *

class YOLO_ACN(nn.Module):
    def __init__(self, classes, anchors):
        super().__init__()
        self.anchors = anchors
        self.classes = classes
        self.backbone = BackBone()
        self.neck = neck()
        self.head = head(classes, anchors)

    def _process_output(self, x):
        low_level_features = x[0]
        mid_level_features = x[1]
        high_level_features = x[2]

        # first 8 is responsible for the first box and so on
        # First row of the first four channels give bbox coordinates
        # 5th channel is objectness
        # 6th channel gives class scores
        # Reshape to to get the form N num_boxes H W Predictions for each anchor box

        high_level_features = einops.rearrange(high_level_features, \
                                               "N C H W -> N (num_boxes) H W P", num_boxes = self.anchors)
        
        bbox = high_level_features[..., :4] # Bbox coordinated for each box
        objectness_score = high_level_features[..., 4:5] # Bbox coordinated for each box
        classification_score = high_level_features[..., 5:] # Bbox coordinated for each box
        



    
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