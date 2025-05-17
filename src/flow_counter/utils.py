from typing import Tuple
import numpy as np

Point = Tuple[int, int]

VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}

def intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """
    Determines whether two line segments AB and CD intersect.

    :param a: Start point of line segment AB
    :param b: End point of line segment AB
    :param c: Start point of line segment CD
    :param d: End point of line segment CD
    :return: True if the segments intersect, False otherwise
    """
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

def ccw(a: Point, b: Point, c: Point) -> bool:
    """
    Determines whether three points a, b, and c are arranged in counter-clockwise order.
    This is calculated using the cross product of vectors.

    :param a: First point
    :param b: Second point
    :param c: Third point
    :return: True if the points are in counter-clockwise order, False otherwise
    """
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU (Intersection over Union) between two boxes.

    :param box1, box2: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area
