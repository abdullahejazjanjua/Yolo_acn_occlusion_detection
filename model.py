from libs import *
from model.head import *
from model.neck import *
from model.backbone import *

class YOLO_ACN(nn.Module):
    def __init__(self, classes, anchors, training=True):
        super().__init__()
        self.anchors = anchors
        self.number_of_anchors = 3
        self.classes = classes
        self.training = training
        self.backbone = BackBone()
        self.neck = neck()
        self.head = head(classes, self.number_of_anchors)

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
                                                "N (num_boxes P) H W -> N (num_boxes) H W P", num_boxes = self.number_of_anchors)
            mid_level_features = einops.rearrange(mid_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes) H W P", num_boxes = self.number_of_anchors)
            low_level_features = einops.rearrange(low_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes) H W P", num_boxes = self.number_of_anchors)
        else:
            # The feature map has a grid of 13 rows and 13 columns, which gives you a total of 169
            # Since, there are anchor boxes. So, we would get 169 * anchor boxes.

            high_level_features = einops.rearrange(high_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes H W) P", num_boxes = self.number_of_anchors)
            mid_level_features = einops.rearrange(mid_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes H W) P", num_boxes = self.number_of_anchors)
            low_level_features = einops.rearrange(low_level_features, \
                                                "N (num_boxes P) H W -> N (num_boxes H W) P", num_boxes = self.number_of_anchors)

        
        hll_bbox = high_level_features[..., :4] # Bbox coordinated for each box
        hll_objectness_score = high_level_features[..., 4:5] # Objectness score for each box
        hll_classification_score = high_level_features[..., 5:] # classification score for each box


        mll_bbox = mid_level_features[..., :4] # Bbox coordinated for each box
        mll_objectness_score = mid_level_features[..., 4:5] # Objectness score for each box
        mll_classification_score = mid_level_features[..., 5:] # classification score for each box

        lll_bbox = low_level_features[..., :4] # Bbox coordinated for each box
        lll_objectness_score = low_level_features[..., 4:5] # Objectness score for each box
        lll_classification_score = low_level_features[..., 5:] # classification score for each box



        return {
            "high_level" : (hll_bbox, hll_objectness_score, hll_classification_score),
            "mid_level": (mll_bbox , mll_objectness_score, mll_classification_score),
            "low_level":  (lll_bbox, lll_objectness_score, lll_classification_score),
        }
    
    def transform_outputs(self, predictions, anchors):
        
        grid_size_hl = predictions["high_level"][0].shape[2]
        grid_size_ml = predictions["mid_level"][0].shape[2]
        grid_size_ll = predictions["low_level"][0].shape[2]


        #creating a 2D grid of coordinates and returning the row and column indices (or coordinates) for each point in the grid
        hl_grid_x, hl_grid_y = torch.meshgrid(torch.arange(grid_size_hl), torch.arange(grid_size_hl), indexing='ij')
        ml_grid_x, ml_grid_y = torch.meshgrid(torch.arange(grid_size_ml), torch.arange(grid_size_ml), indexing='ij')
        ll_grid_x, ll_grid_y = torch.meshgrid(torch.arange(grid_size_ll), torch.arange(grid_size_ll), indexing='ij')


        hl_grid_x = einops.rearrange(hl_grid_x, "H W -> 1 1 H  W")
        hl_grid_y = einops.rearrange(hl_grid_y, "H W -> 1 1 H  W")
        ml_grid_x = einops.rearrange(ml_grid_x, "H W -> 1 1 H  W")
        ml_grid_y = einops.rearrange(ml_grid_y, "H W -> 1 1 H  W")
        ll_grid_x = einops.rearrange(ll_grid_x, "H W -> 1 1 H  W")
        ll_grid_y = einops.rearrange(ll_grid_y, "H W -> 1 1 H  W")

        hl_tx, hl_ty, hl_tw, hl_th = predictions["high_level"][0][:, :, :, :, 0] , predictions["high_level"][0][:, :, :, :, 1], predictions["high_level"][0][:, :, :, :, 2], predictions["high_level"][0][:, :, :, :, 3]
        ml_tx, ml_ty, ml_tw, ml_th = predictions["mid_level"][0][:, :, :, :, 0] , predictions["mid_level"][0][:, :, :, :, 1], predictions["mid_level"][0][:, :, :, :, 2], predictions["mid_level"][0][:, :, :, :, 3]
        ll_tx, ll_ty, ll_tw, ll_th = predictions["low_level"][0][:, :, :, :, 0] , predictions["low_level"][0][:, :, :, :, 1], predictions["low_level"][0][:, :, :, :, 2], predictions["low_level"][0][:, :, :, :, 3]  


        hl_cx = F.sigmoid(hl_tx) + hl_grid_x.float()
        hl_cy = F.sigmoid(hl_ty) + hl_grid_y.float()
        ml_cx = F.sigmoid(ml_tx) + ml_grid_x.float()
        ml_cy = F.sigmoid(ml_ty) + ml_grid_y.float()
        ll_cx = F.sigmoid(ll_tx) + ll_grid_x.float()
        ll_cy = F.sigmoid(ll_ty) + ll_grid_y.float()


        high_level_anchors = torch.tensor(anchors[:3])  # First three anchors
        mid_level_anchors = torch.tensor(anchors[3:6])  # Next three anchors
        low_level_anchors = torch.tensor(anchors[6:])   # Last three anchors

        hl_anchor_height, hl_anchor_width = high_level_anchors[:,0], high_level_anchors[:, 1]
        ml_anchor_height, ml_anchor_width = mid_level_anchors[:, 0], mid_level_anchors[:, 1]
        ll_anchor_height, ll_anchor_width = low_level_anchors[:, 0], low_level_anchors[:, 1]

        hl_anchor_height = einops.rearrange(hl_anchor_height, "A -> 1 A 1 1")
        hl_anchor_width = einops.rearrange(hl_anchor_width, "A -> 1 A 1 1")

        ml_anchor_height = einops.rearrange(ml_anchor_height, "A -> 1 A 1 1")
        ml_anchor_width = einops.rearrange(ml_anchor_width, "A -> 1 A 1 1")

        ll_anchor_height = einops.rearrange(ll_anchor_height, "A -> 1 A 1 1")
        ll_anchor_width = einops.rearrange(ll_anchor_width, "A -> 1 A 1 1")



        hl_h = torch.exp(hl_th) * hl_anchor_height
        hl_w = torch.exp(hl_tw) * hl_anchor_width
        
        ml_h = torch.exp(ml_th) * ml_anchor_height
        ml_w = torch.exp(ml_tw) * ml_anchor_width
    
        ll_h = torch.exp(ll_th) * ll_anchor_height
        ll_w = torch.exp(ll_tw) * ll_anchor_width


        predictions["high_level"] = (torch.stack([hl_cx, hl_cy, hl_h, hl_w], dim=-1), predictions['high_level'][1], predictions['high_level'][2])
        predictions["mid_level"] = (torch.stack([ml_cx, ml_cy, ml_h, ml_w], dim=-1), predictions['mid_level'][1], predictions['mid_level'][2])
        predictions["low_level"] = (torch.stack([ll_cx, ll_cy, ll_h, ll_w], dim=-1), predictions['low_level'][1], predictions['low_level'][2])

        return predictions


    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        x = self._process_output(x)
        x = self.transform_outputs(x, self.anchors)

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

    print(f"High level Shape: {x["high_level"][0].shape}")
    print(f"Mid level Shape: {x["mid_level"][0].shape}")
    print(f"low level Shape: {x["low_level"][0].shape}")
