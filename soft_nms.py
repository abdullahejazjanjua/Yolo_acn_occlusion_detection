from loss.metric import *
from libs import *

def gaussian_decay(iou_diff, sigma=0.5):
    
    decay = torch.exp(-(iou_diff ** 2) / (2 * sigma ** 2))
    return decay

def apply_nms(predictions, iou_thres, Nt=0.3):

    '''
    If IoU - CIoU <= Nt, Si
        
    if IoU - CIoU > Nt then decay Si using Gaussian decay
    '''
    with torch.no_grad():
        for level in predictions:

            bboxes, objectness_score, classification_score = predictions[level]

                
            final_score = objectness_score * classification_score
            keep = []

            while len(final_score) > 0:
                
                max_idx = torch.argmax(final_score, dim=-1)  # TODO: FIX THIS :-(
                print(max_idx.shape)
                bbox_max = bboxes[max_idx]
                keep.append(max_idx)

                CIoU = compute_CIoU(bboxes, bbox_max)
                IoU = Compute_IoU(bboxes, bbox_max)
                diff = CIoU - IoU

                decay = torch.where(diff > Nt, gaussian_decay(diff), torch.ones_like(final_score)) # if diff > Nt, select gaussian decay else set to 1
                final_score = final_score * decay

                mask = IoU < iou_thres
                bboxes = bboxes[mask]
                final_score = final_score[mask]

                

        predictions[level] = (bboxes, final_score)

    return predictions

if __name__ == "__main__":

        
    predictions = {
    'high_level': (torch.rand(2, 3, 13, 13, 4), torch.rand(2, 3, 13, 13, 1), torch.rand(2, 3, 13, 13, 3)),
    'mid_level': (torch.rand(2, 3, 13, 13, 4), torch.rand(2, 3, 13, 13, 1), torch.rand(2, 3, 13, 13, 3)),
    'low_level': (torch.rand(2, 3, 13, 13, 4), torch.rand(2, 3, 13, 13, 1), torch.rand(2, 3, 13, 13, 3))
    }
    
    iou_threshold = 0.5
    Nt = 0.3
    predictions = apply_nms(predictions, iou_threshold, Nt)

    print(predictions)
