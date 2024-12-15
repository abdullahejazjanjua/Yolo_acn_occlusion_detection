import torch
def Compute_IoU(bboxes, grd_truth, eps=1e-8):

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

    return (intersection_area / (Union_area + eps)) , (x_max_pred, y_max_pred, x_max_grd, y_max_grd, x_min_pred, y_min_pred, x_min_grd, y_min_grd)

def compute_CIoU(bboxes, grd_truth, eps=1e-8):
    cx_pred, cy_pred, width_pred, height_pred = bboxes[...,0], bboxes[...,1], bboxes[...,2], bboxes[...,3]
    cx_grd, cy_grd, width_grd, height_grd = grd_truth[...,0], grd_truth[...,1], grd_truth[...,2], grd_truth[...,3]

    p = ((cx_pred - cx_grd)**2) + ((cy_pred - cy_grd)**2)
        
    v = (4 / (torch.pi**2)) * torch.pow((torch.atan(width_grd / height_grd) - torch.atan(width_pred / width_grd)), 2)


    IoU, coordinates = Compute_IoU(bboxes, grd_truth)

    x_max_pred, y_max_pred, x_max_grd, y_max_grd, x_min_pred, y_min_pred, x_min_grd, y_min_grd = coordinates

    x_min_enclosing = torch.min(x_min_grd, x_min_pred)
    y_min_enclosing = torch.min(y_min_grd, y_min_pred)
    x_max_enclosing = torch.max(x_max_grd, x_max_pred)
    y_max_enclosing = torch.max(y_max_grd, y_max_pred)

    enclosing_x = x_max_enclosing - x_min_enclosing
    enclosing_y = y_max_enclosing - y_min_enclosing
    c = torch.sqrt(enclosing_x**2 + enclosing_y**2)
    with torch.no_grad():
        alpha_ = v / (1 - IoU + v + eps)
    R_DIoU = p / (c + eps)

    CIoU = 1 - IoU + R_DIoU + alpha_ * v        

    return CIoU