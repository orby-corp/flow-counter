from typing import Tuple

Point = Tuple[int, int]

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
