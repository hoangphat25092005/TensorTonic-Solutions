import numpy as np

def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    box format: [x1, y1, x2, y2]
    """

    box_a = np.array(box_a)
    box_b = np.array(box_b)

    # area of boxes
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # intersection
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    intersection = inter_width * inter_height

    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union