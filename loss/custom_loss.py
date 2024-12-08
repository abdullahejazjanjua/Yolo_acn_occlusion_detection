from libs import *

class YOLO_ACN_Loss():
    def __init__(self, IoU_threshold, eps=1e-8):
        self.IoU_threshold = IoU_threshold
        self.eps = eps

    def Compute_IoU(self, bboxes, grd_truth):

        # Extract the Predictions and Ground Truth
        x_min_pred, y_min_pred, width_pred, height_pred = bboxes[...,0], bboxes[...,1], bboxes[...,2], bboxes[...,3]
        x_min_grd, y_min_grd, width_grd, height_grd = grd_truth[...,0], grd_truth[...,1], grd_truth[...,2], grd_truth[...,3]

        # Calculating x_max, y_max for predictions
        x_max_pred = x_min_pred + width_pred
        y_max_pred = y_min_pred + height_pred

        # Calculating x_max, y_max for ground truth
        x_max_grd = x_min_grd + width_grd
        y_max_grd = y_min_grd + height_grd

        # Calculating the intersection Coordinates
        x_min_intersection = torch.max(x_min_grd, x_min_grd)
        y_min_intersection = torch.max(y_min_grd, y_min_grd)
        x_max_intersection = torch.max(x_max_grd, x_max_grd)
        y_max_intersection = torch.max(y_max_grd, y_max_grd)

        # Calculating the Intersection Width and Height
        intersection_width = torch.clamp(x_max_intersection - x_min_intersection, min=0)
        intersection_height = torch.clamp(y_max_intersection - y_min_intersection, min=0)


        intersection_area = intersection_width * intersection_height

        area_grd = width_grd * height_grd
        area_pred = width_pred * height_pred

        Union_area = area_grd + area_pred - intersection_area

        return intersection_area / (Union_area + self.eps)

    def compute_CIoU(self, bboxes, grd_truth):
        pass
        
    def forward(self, predictions, ground_truth):

        total_loss = 0
        for level_predictions in predictions:

            bboxes, objectness_score, classification_scores = level_predictions[0], level_predictions[1], level_predictions[2]
            bboxes_grd, objectness_score_grd, classification_scores_grd = ground_truth[0], ground_truth[1], ground_truth[2]

            CIoU = self.compute_CIoU(bboxes, bboxes_grd)

            objectness_loss = F.binary_cross_entropy_with_logits(objectness_score_grd, objectness_score)
            classification_loss = F.cross_entropy(classification_scores, classification_scores_grd)

            total_loss += CIoU + objectness_loss + classification_loss

        
        return total_loss
        
        
             