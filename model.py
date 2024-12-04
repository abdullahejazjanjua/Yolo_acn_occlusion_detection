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
                                               "N (num_boxes P) H W -> N (num_boxes) H W P", num_boxes = self.anchors)
        mid_level_features = einops.rearrange(mid_level_features, \
                                               "N (num_boxes P) H W -> N (num_boxes) H W P", num_boxes = self.anchors)
        low_level_features = einops.rearrange(low_level_features, \
                                               "N (num_boxes P) H W -> N (num_boxes) H W P", num_boxes = self.anchors)
        
        hll_bbox = high_level_features[..., :4] # Bbox coordinated for each box
        hll_objectness_score = high_level_features[..., 4:5] # Bbox coordinated for each box
        hll_classification_score = high_level_features[..., 5:] # Bbox coordinated for each box


        mll_bbox = mid_level_features[..., :4] # Bbox coordinated for each box
        mll_objectness_score = mid_level_features[..., 4:5] # Bbox coordinated for each box
        mll_classification_score = mid_level_features[..., 5:] # Bbox coordinated for each box

        lll_bbox = low_level_features[..., :4] # Bbox coordinated for each box
        lll_objectness_score = low_level_features[..., 4:5] # Bbox coordinated for each box
        lll_classification_score = low_level_features[..., 5:] # Bbox coordinated for each box

        Predictions = namedtuple('Predictions',['high_level', 'mid_level', 'low_level'])


        return Predictions(
            high_level = (hll_bbox, hll_objectness_score, hll_classification_score),
            mid_level = (mll_bbox , mll_objectness_score, mll_classification_score),
            low_level = (lll_bbox, lll_objectness_score, lll_classification_score),
        )

    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        x = self._process_output(x)

        return x


if __name__ == "__main__":
    x = torch.randn((2, 3, 416, 416))
    model = YOLO_ACN(80, 3)
    x = model(x)
    print(f"High level bbox for 3 Anchor boxes: {x.high_level[0].shape}")
    print(f"High level objectness score for 3 Anchor boxes: {x.high_level[1].shape}")
    print(f"High level classification for 3 Anchor boxes: {x.high_level[2].shape}")
