from typing import Tuple
import cv2
import numpy as np

Point = Tuple[int, int]

VEHICLE_NAMES = {"person", "car", "motorcycle", "bus", "truck"}

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

def draw_table_on_image(
    image: np.ndarray,
    table_data: list[list[str]],
    start_x: int = 10,
    start_y: int = 10,
    cell_width: int = 150,
    cell_height: int = 40,
    font_scale: float = 0.6,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw a table on the top-left corner of an image using OpenCV.

    Args:
        image (np.ndarray): The target image to draw on.
        table_data (list[list[str]]): A 2D list representing table data, 
                                      e.g., [["Header1", "Header2"], ["Row1Col1", "Row1Col2"], ...].
        start_x (int): X coordinate of the top-left corner of the table.
        start_y (int): Y coordinate of the top-left corner of the table.
        cell_width (int): Width of each cell in pixels.
        cell_height (int): Height of each cell in pixels.
        font_scale (float): Font scale for text.
        thickness (int): Thickness for rectangle borders and text.

    Returns:
        np.ndarray: The image with the table drawn on it.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            top_left = (start_x + j * cell_width, start_y + i * cell_height)
            bottom_right = (start_x + (j + 1) * cell_width, start_y + (i + 1) * cell_height)
            # Fill cell background with white
            cv2.rectangle(image, top_left, bottom_right, color=(255, 255, 255), thickness=-1)
            # Draw cell border in black
            cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 0), thickness=thickness)
            # Center the text in the cell
            text_size = cv2.getTextSize(cell, font, font_scale, thickness)[0]
            text_x = top_left[0] + (cell_width - text_size[0]) // 2
            text_y = top_left[1] + (cell_height + text_size[1]) // 2
            cv2.putText(image, cell, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return image