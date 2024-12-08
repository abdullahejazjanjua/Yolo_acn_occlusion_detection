from libs import *
from model.head import *
from model.neck import *
from model.backbone import *

class YOLO_ACN(nn.Module):
    def __init__(self, classes, anchors, training=True):
        super().__init__()
        self.anchors = anchors
        self.number_of_anchors = len(anchors)
        self.classes = classes
        self.training = training
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
        if self.training:
            high_level_features = einops.rearrange(high_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes) H W P", num_boxes = self.number_of_anchorsanchors)
            mid_level_features = einops.rearrange(mid_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes) H W P", num_boxes = self.number_of_anchorsanchors)
            low_level_features = einops.rearrange(low_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes) H W P", num_boxes = self.number_of_anchorsanchors)
        else:
            # The feature map has a grid of 13 rows and 13 columns, which gives you a total of 169
            # Since, there are anchor boxes. So, we would get 169 * anchor boxes.

            high_level_features = einops.rearrange(high_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes H W) P", num_boxes = self.number_of_anchorsanchors)
            mid_level_features = einops.rearrange(mid_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes H W) P", num_boxes = self.number_of_anchorsanchors)
            low_level_features = einops.rearrange(low_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes H W) P", num_boxes = self.number_of_anchorsanchors)

        
        hll_bbox = high_level_features[..., :4] # Bbox coordinated for each box
        hll_objectness_score = high_level_features[..., 4:5] # Objectness score for each box
        hll_classification_score = high_level_features[..., 5:] # classification score for each box


        mll_bbox = mid_level_features[..., :4] # Bbox coordinated for each box
        mll_objectness_score = mid_level_features[..., 4:5] # Objectness score for each box
        mll_classification_score = mid_level_features[..., 5:] # classification score for each box

        lll_bbox = low_level_features[..., :4] # Bbox coordinated for each box
        lll_objectness_score = low_level_features[..., 4:5] # Objectness score for each box
        lll_classification_score = low_level_features[..., 5:] # classification score for each box

        Predictions = namedtuple('Predictions',['high_level', 'mid_level', 'low_level'])


        return Predictions(
            high_level = (hll_bbox, hll_objectness_score, hll_classification_score),
            mid_level = (mll_bbox , mll_objectness_score, mll_classification_score),
            low_level = (lll_bbox, lll_objectness_score, lll_classification_score),
        )
    
    def transform_outputs(self, predictions):
        pass



    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        x = self._process_output(x)

        return x


if __name__ == "__main__":

    anchors = [
    (116, 90), (156, 198), (373, 326),  # anchor box 1 dimensions
    (30, 61), (62, 45), (59, 119),    # anchor box 2 dimensions
    (10, 13), (16, 30), (33, 23),     # anchor box 3 dimensions
    ]

    number_of_anchors = len(anchors)


    x = torch.randn((2, 3, 416, 416))
    model = YOLO_ACN(classes=3, anchors=anchors)
    x = model(x)

    print(f"High level Shape: {x.high_level[0].shape}")
    print(f"Mid level Shape: {x.mid_level[0].shape}")
    print(f"low level Shape: {x.low_level[0].shape}")
