import numpy as np
from flow_counter.utils import compute_iou

def test_iou_perfect_overlap():
    box = np.array([10, 10, 20, 20])
    assert compute_iou(box, box) == 1.0

def test_iou_no_overlap():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([20, 20, 30, 30])
    assert compute_iou(box1, box2) == 0.0

def test_iou_partial_overlap():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([5, 5, 15, 15])
    # Intersection: (5,5)-(10,10) = 25, union: 100+100-25=175
    expected = 25 / 175
    assert np.isclose(compute_iou(box1, box2), expected)

def test_iou_edge_touching():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([10, 0, 20, 10])
    # Edge just touches, no overlap
    assert compute_iou(box1, box2) == 0.0

def test_iou_contained():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([2, 2, 8, 8])
    # box2 is completely inside box1
    inter = (8-2)*(8-2) # 36
    area1 = 100
    area2 = 36
    expected = inter / (area1 + area2 - inter)
    assert np.isclose(compute_iou(box1, box2), expected)
