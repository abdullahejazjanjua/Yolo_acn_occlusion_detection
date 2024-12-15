import torch.nn.functional as F
from metric import *  

class YOLO_ACN_Loss():
    def __init__(self, IoU_threshold, eps=1e-8):
        self.IoU_threshold = IoU_threshold
        self.eps = eps
        

            
    def forward(self, predictions, ground_truth):

        total_loss = 0
        for level_predictions in predictions:

            bboxes, objectness_score, classification_scores = predictions[level_predictions]
            bboxes_grd, objectness_score_grd, classification_scores_grd = ground_truth[0], ground_truth[1], ground_truth[2]

            CIoU = compute_CIoU(bboxes, bboxes_grd)

            objectness_loss = F.binary_cross_entropy_with_logits(objectness_score_grd, objectness_score)
            classification_loss = F.cross_entropy(classification_scores, classification_scores_grd) # TODO: Re Shape it when training 
            
            total_loss += CIoU.mean() + objectness_loss + classification_loss

        
        return total_loss
        

if __name__ == "__main__":
    import torch
    loss_fn = YOLO_ACN_Loss(IoU_threshold=0.5)

    # Predictions (center_x, center_y, width, height)
    bboxes_pred = torch.rand((2, 3, 13, 13, 4))      
    bboxes_grd = torch.rand((2, 3, 13, 13, 4))

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
