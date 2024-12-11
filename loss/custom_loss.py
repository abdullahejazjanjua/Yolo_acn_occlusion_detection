import torch.nn.functional as F  
import torchvision.ops as to

class YOLO_ACN_Loss():
    def __init__(self, IoU_threshold, eps=1e-8):
        self.IoU_threshold = IoU_threshold
        self.eps = eps
        

    def Compute_IoU(self, bboxes, grd_truth):

        # Extract the Predictions and Ground Truth
        cx_pred, cy_pred, width_pred, height_pred = bboxes[...,0], bboxes[...,1], bboxes[...,2], bboxes[...,3]
        cx_grd, cy_grd, width_grd, height_grd = grd_truth[...,0], grd_truth[...,1], grd_truth[...,2], grd_truth[...,3]

        # Calculating x_max, y_max for predictions
        x_max_pred = cx_pred + width_pred / 2
        y_max_pred = cy_pred + height_pred / 2

        # Calculating x_min, y_min for predictions
        x_min_pred = cx_pred - width_pred / 2  
        y_min_pred = cy_pred - height_pred / 2

        # Calculating x_max, y_max for ground truth
        x_max_grd = cx_grd + width_grd / 2
        y_max_grd = cy_grd + height_grd / 2

        
        # Calculating x_min, y_min for ground truth
        x_min_grd = cx_grd - width_grd / 2
        y_min_grd = cy_grd - height_grd / 2

        # Calculating the intersection Coordinates
        x_min_intersection = torch.max(x_min_grd, x_min_pred)
        y_min_intersection = torch.max(y_min_grd, y_min_pred)
        x_max_intersection = torch.min(x_max_grd, x_max_pred)
        y_max_intersection = torch.min(y_max_grd, y_max_pred)

        # Calculating the Intersection Width and Height
        intersection_width = torch.clamp(x_max_intersection - x_min_intersection, min=0)
        intersection_height = torch.clamp(y_max_intersection - y_min_intersection, min=0)


        intersection_area = intersection_width * intersection_height

        area_grd = width_grd * height_grd
        area_pred = width_pred * height_pred

        Union_area = area_grd + area_pred - intersection_area

        return (intersection_area / (Union_area + self.eps)) , (x_max_pred, y_max_pred, x_max_grd, y_max_grd, x_min_pred, y_min_pred, x_min_grd, y_min_grd)

    def compute_CIoU(self, bboxes, grd_truth):
        cx_pred, cy_pred, width_pred, height_pred = bboxes[...,0], bboxes[...,1], bboxes[...,2], bboxes[...,3]
        cx_grd, cy_grd, width_grd, height_grd = grd_truth[...,0], grd_truth[...,1], grd_truth[...,2], grd_truth[...,3]

        p = ((cx_pred - cx_grd)**2) + ((cy_pred - cy_grd)**2)
        

        v = 4/torch.pi * (
            (torch.arctan(width_grd / height_grd + self.eps) - torch.arctan(width_pred / height_pred + self.eps)) **2
        )


        IoU, coordinates = self.Compute_IoU(bboxes, grd_truth)

        x_max_pred, y_max_pred, x_max_grd, y_max_grd, x_min_pred, y_min_pred, x_min_grd, y_min_grd = coordinates

        x_min_enclosing = torch.min(x_min_grd, x_min_pred)
        y_min_enclosing = torch.min(y_min_grd, y_min_pred)
        x_max_enclosing = torch.max(x_max_grd, x_max_pred)
        y_max_enclosing = torch.max(y_max_grd, y_max_pred)

        enclosing_x = x_max_enclosing - x_min_enclosing
        enclosing_y = y_max_enclosing - y_min_enclosing
        c = torch.sqrt(enclosing_x**2 + enclosing_y**2)

        alpha_ = v / ((1 - IoU) + v)
        R_DIoU = p / (c + self.eps)

        CIoU = 1 - IoU + R_DIoU + alpha_ * v
        # Combine individual tensors into (N, 4) format
        bboxes_pred_02 = torch.stack([x_min_pred, y_min_pred, x_max_pred, y_max_pred], dim=-1)  # Shape: (N, 4)
        bboxes_grd_02 = torch.stack([x_min_grd, y_min_grd, x_max_grd, y_max_grd], dim=-1)      # Shape: (N, 4)

        # Call the function with properly formatted inputs
        CIoU_02 = to.complete_box_iou_loss(bboxes_pred_02, bboxes_grd_02)


        

        print(f"CIoU: Min {torch.min(CIoU)}, Max {torch.max(CIoU)}, Mean {CIoU.mean()}")
        print(f"CIoU_02: Min {torch.min(CIoU_02)}, Max {torch.max(CIoU_02)}, Mean {CIoU_02.mean()}")        
        

        return CIoU
                
    def forward(self, predictions, ground_truth):

        total_loss = 0
        for level_predictions in predictions:

            bboxes, objectness_score, classification_scores = predictions[level_predictions]
            bboxes_grd, objectness_score_grd, classification_scores_grd = ground_truth[0], ground_truth[1], ground_truth[2]

            CIoU = self.compute_CIoU(bboxes, bboxes_grd)

            objectness_loss = F.binary_cross_entropy_with_logits(objectness_score_grd, objectness_score)
            classification_loss = F.cross_entropy(classification_scores, classification_scores_grd)
            
            total_loss += CIoU.mean() + objectness_loss + classification_loss

        
        return total_loss
        

if __name__ == "__main__":
    import torch
    loss_fn = YOLO_ACN_Loss(IoU_threshold=0.5)

    # Predictions (center_x, center_y, width, height)
    bboxes_pred = torch.rand((2, 3, 13, 13, 4))      
    bboxes_grd = torch.rand((2, 3, 13, 13, 4))
    print(bboxes_pred.min(), bboxes_pred.max())
    print(bboxes_grd.min(), bboxes_grd.max())

    # Objectness scores 
    objectness_pred = torch.randn((2, 3, 13, 13, 1))
    objectness_grd = torch.randint(0, 2, (2, 3, 13, 13, 1)).float()

    # classification scores
    classification_scores = torch.randn(2, 3, 13, 13, 3) 
    classification_scores_grd = torch.randint(0, 3, (2, 3, 13, 13))  

    classification_scores = classification_scores.view(-1, 3)  
    classification_scores_grd = classification_scores_grd.view(-1)  
    predictions = {
    0: (bboxes_pred, objectness_pred, classification_scores)
    }
    ground_truth = (bboxes_grd, objectness_grd, classification_scores_grd)
    total_loss = loss_fn.forward(predictions, ground_truth)
    print("Total Loss:", total_loss)
