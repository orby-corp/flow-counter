import numpy as np
from pytest_mock import MockerFixture

from flow_counter import FlowCounter
from flow_counter.utils import Point

LINE = tuple[Point, Point]

def test_crossed_lines_merged_when_root_switches_to_box_id1_side(
    mocker: MockerFixture,
    dummy_two_lines: dict[str, tuple[LINE, LINE]],
    flow_counter: FlowCounter,
) -> None:
    """
    Reproduces issue #1 cause 5: when a tracker ID switch causes unite() to
    change the root ID, crossed_lines history recorded under the old root
    must still be visible under the new root.

    Frame 1: object "A" crosses line1 only.
    Frame 2: a new ID "B" appears at the same location (IoU=1.0 with "A"),
    and "B" crosses line2. unite("B", "A") is called; since both sides have
    equal union-find size, "B" (box_id1) remains the surviving root, so
    crossed_lines["A"] must be merged into crossed_lines["B"].
    """
    flow_counter.model.names = {0: "car"}

    # Frame 1: "A" crosses line1 only.
    boxes_frame1 = np.array([[10, 10, 20, 20]])
    ids_frame1 = np.array(["A"])
    classes_frame1 = np.array([0])
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[True, False])
    count1 = flow_counter._count_crossing_objects(boxes_frame1, ids_frame1, classes_frame1, dummy_two_lines)
    assert count1 == 0

    # Frame 2: "A" and "B" occupy the same box (IoU=1.0); "B" crosses line2.
    boxes_frame2 = np.array([[10, 10, 20, 20], [10, 10, 20, 20]])
    ids_frame2 = np.array(["A", "B"])
    classes_frame2 = np.array([0, 0])
    # intersect is checked per box in order: A(line1, line2), B(line1, line2)
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, False, False, True])
    count2 = flow_counter._count_crossing_objects(boxes_frame2, ids_frame2, classes_frame2, dummy_two_lines)

    assert count2 == 1
    assert flow_counter.cls_counts["car"]["dummy"] == 1

def test_crossed_lines_merged_when_root_switches_to_box_id2_side(
    mocker: MockerFixture,
    dummy_two_lines: dict[str, tuple[LINE, LINE]],
    flow_counter: FlowCounter,
) -> None:
    """
    Mirror image of the test above: this time box_id1 ("A", holding the
    pre-existing history) is the one that gets swallowed, and box_id2's
    side ("B", pre-merged with "D" to size 2) survives as the new root.
    The merge must still work when new_root == root_id2 instead of
    root_id1.
    """
    flow_counter.model.names = {0: "car"}

    # Pre-merge "B" with "D" so root("B")'s cluster size is 2, larger than
    # "A" (size 1), forcing unite("A", root("B")) to keep "B" as root
    # instead of "A".
    flow_counter.uf.unite("B", "D")

    # Frame 1: "A" crosses line1 only.
    boxes_frame1 = np.array([[10, 10, 20, 20]])
    ids_frame1 = np.array(["A"])
    classes_frame1 = np.array([0])
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[True, False])
    flow_counter._count_crossing_objects(boxes_frame1, ids_frame1, classes_frame1, dummy_two_lines)

    # Frame 2: "A" and "B" occupy the same box (IoU=1.0); "A" now crosses line2.
    boxes_frame2 = np.array([[10, 10, 20, 20], [10, 10, 20, 20]])
    ids_frame2 = np.array(["A", "B"])
    classes_frame2 = np.array([0, 0])
    # intersect is checked per box in order: A(line1, line2), B(line1, line2)
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, True, False, False])
    count2 = flow_counter._count_crossing_objects(boxes_frame2, ids_frame2, classes_frame2, dummy_two_lines)

    assert count2 == 1
    assert flow_counter.cls_counts["car"]["dummy"] == 1

def test_crossed_lines_merge_does_not_double_count(
    mocker: MockerFixture,
    dummy_two_lines: dict[str, tuple[LINE, LINE]],
    flow_counter: FlowCounter,
) -> None:
    """
    Once an object has been counted, a later ID-switch merge involving its
    root must not trigger a second count.
    """
    flow_counter.model.names = {0: "car"}

    # Frame 1: "A" crosses both lines and gets counted.
    boxes_frame1 = np.array([[10, 10, 20, 20]])
    ids_frame1 = np.array(["A"])
    classes_frame1 = np.array([0])
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[True, True])
    count1 = flow_counter._count_crossing_objects(boxes_frame1, ids_frame1, classes_frame1, dummy_two_lines)
    assert count1 == 1

    # Frame 2: a new ID "B" appears at the same location and also crosses a line.
    boxes_frame2 = np.array([[10, 10, 20, 20], [10, 10, 20, 20]])
    ids_frame2 = np.array(["A", "B"])
    classes_frame2 = np.array([0, 0])
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, False, True, False])
    count2 = flow_counter._count_crossing_objects(boxes_frame2, ids_frame2, classes_frame2, dummy_two_lines)

    assert count2 == 0
    assert flow_counter.cls_counts["car"]["dummy"] == 1
