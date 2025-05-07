from flow_counter.utils import intersect, Point

def test_intersect_with_crossing_lines() -> None:
    """
    Test that two lines clealy crossing return True.
    """
    p1, p2 = (0, 0), (10, 10)
    q1, q2 = (0, 10), (10, 0)
    assert intersect(p1, p2, q1, q2)

def test_intersect_parallel_lines() -> None:
    """
    Test that two parallel lines return False.
    """
    p1, p2 = (0, 0), (10, 0)
    q1, q2 = (0, 1), (10, 1)
    assert not intersect(p1, p2, q1, q2)

def test_intersect_touching_at_endpoint() -> None:
    """
    Test that two lines touching at one endpoint are considered intersecting.
    """
    p1, p2 = (0, 0), (10, 0)
    q1, q2 = (10, 0), (20, 0)
    assert not intersect(p1, p2, q1, q2)

def test_intersect_overlapping_lines() -> None:
    """
    Test that overlapping colinear segments return False.
    """
    p1, p2 = (0, 0), (10, 0)
    q1, q2 = (5, 0), (15, 0)
    assert not intersect(p1, p2, q1, q2)

def test_intersect_no_contact() -> None:
    """
    Test that two far-apart lines return False.
    """
    p1, p2 = (0, 0), (1, 1)
    q1, q2 = (100, 100), (110, 110)
    assert not intersect(p1, p2, q1, q2)