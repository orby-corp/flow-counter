import numpy as np
from pytest_mock import MockerFixture

from flow_counter import FlowCounter
from flow_counter.utils import Point

LINE = tuple[Point, Point]

def test_forward_crossing_is_not_flagged_as_reverse(
    mocker: MockerFixture,
    dummy_two_lines: dict[str, tuple[LINE, LINE]],
    flow_counter: FlowCounter,
) -> None:
    """
    Issue #1 cause 1: an object that crosses line1 then line2 (the intended
    direction) must not be recorded as a reverse crossing.
    """
    flow_counter.model.names = {0: "car"}

    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array(["A"])
    classes = np.array([0])

    # Frame 1: crosses line1 only.
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[True, False])
    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_two_lines)

    # Frame 2: crosses line2.
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, True])
    count = flow_counter._count_crossing_objects(boxes, ids, classes, dummy_two_lines)

    assert count == 1
    assert flow_counter.reverse_crossings.get("dummy", 0) == 0

def test_reverse_crossing_is_flagged_but_still_counted(
    mocker: MockerFixture,
    dummy_two_lines: dict[str, tuple[LINE, LINE]],
    flow_counter: FlowCounter,
) -> None:
    """
    Issue #1 cause 1: an object that crosses line2 before line1 (reverse
    direction) is still counted today, but must now be recorded in
    reverse_crossings so it can be investigated later.
    """
    flow_counter.model.names = {0: "car"}

    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array(["A"])
    classes = np.array([0])

    # Frame 1: crosses line2 only.
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, True])
    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_two_lines)

    # Frame 2: crosses line1.
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[True, False])
    count = flow_counter._count_crossing_objects(boxes, ids, classes, dummy_two_lines)

    assert count == 1
    assert flow_counter.cls_counts["car"]["dummy"] == 1
    assert flow_counter.reverse_crossings["dummy"] == 1

def test_first_crossed_side_survives_root_switch(
    mocker: MockerFixture,
    dummy_two_lines: dict[str, tuple[LINE, LINE]],
    flow_counter: FlowCounter,
) -> None:
    """
    Mirrors test_crossed_lines_merge.py: when a tracker ID switch causes
    unite() to change the root ID, the first_crossed_side record must be
    carried over to the new root so reverse-direction detection still works.

    Frame 1: "A" crosses line2 only (the reverse side first).
    Frame 2: a new ID "B" appears at the same location (IoU=1.0 with "A"),
    triggering unite("B", "A"); "B" then crosses line1.
    """
    flow_counter.model.names = {0: "car"}

    # Frame 1: "A" crosses line2 only.
    boxes_frame1 = np.array([[10, 10, 20, 20]])
    ids_frame1 = np.array(["A"])
    classes_frame1 = np.array([0])
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, True])
    flow_counter._count_crossing_objects(boxes_frame1, ids_frame1, classes_frame1, dummy_two_lines)

    # Frame 2: "A" and "B" occupy the same box (IoU=1.0); "B" crosses line1.
    boxes_frame2 = np.array([[10, 10, 20, 20], [10, 10, 20, 20]])
    ids_frame2 = np.array(["A", "B"])
    classes_frame2 = np.array([0, 0])
    # intersect is checked per box in order: A(line1, line2), B(line1, line2)
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, False, True, False])
    count2 = flow_counter._count_crossing_objects(boxes_frame2, ids_frame2, classes_frame2, dummy_two_lines)

    assert count2 == 1
    assert flow_counter.cls_counts["car"]["dummy"] == 1
    assert flow_counter.reverse_crossings["dummy"] == 1
