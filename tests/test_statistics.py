import numpy as np
from pytest_mock import MockerFixture

from flow_counter import FlowCounter
from flow_counter.utils import Point

LINE = tuple[Point, Point]

def test_crossing_history_records_frame_and_details(
    mocker: MockerFixture,
    dummy_two_lines: dict[str, tuple[LINE, LINE]],
    flow_counter: FlowCounter,
) -> None:
    flow_counter.model.names = {0: "car"}

    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array(["A"])
    classes = np.array([0])

    # Frame 1: crosses line1 only.
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[True, False])
    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_two_lines)

    # Frame 2: crosses line2, completing the pass.
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, True])
    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_two_lines)

    assert flow_counter.crossing_history == [{
        "frame": 2,
        "root_id": "A",
        "class_name": "car",
        "line_name": "dummy",
        "reverse": False,
    }]

def test_crossing_history_flags_reverse_crossings(
    mocker: MockerFixture,
    dummy_two_lines: dict[str, tuple[LINE, LINE]],
    flow_counter: FlowCounter,
) -> None:
    flow_counter.model.names = {0: "car"}

    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array(["A"])
    classes = np.array([0])

    # Frame 1: crosses line2 first (reverse direction).
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, True])
    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_two_lines)

    # Frame 2: crosses line1, completing the pass.
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[True, False])
    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_two_lines)

    assert flow_counter.crossing_history[0]["frame"] == 2
    assert flow_counter.crossing_history[0]["reverse"] is True

def test_merge_history_records_frame_and_roots(
    mocker: MockerFixture,
    dummy_two_lines: dict[str, tuple[LINE, LINE]],
    flow_counter: FlowCounter,
) -> None:
    flow_counter.model.names = {0: "car"}

    # Frame 1: "A" crosses line1 only.
    boxes_frame1 = np.array([[10, 10, 20, 20]])
    ids_frame1 = np.array(["A"])
    classes_frame1 = np.array([0])
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[True, False])
    flow_counter._count_crossing_objects(boxes_frame1, ids_frame1, classes_frame1, dummy_two_lines)

    # Frame 2: "A" and "B" occupy the same box (IoU=1.0); "B" crosses line2,
    # triggering unite("B", "A").
    boxes_frame2 = np.array([[10, 10, 20, 20], [10, 10, 20, 20]])
    ids_frame2 = np.array(["A", "B"])
    classes_frame2 = np.array([0, 0])
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, False, False, True])
    flow_counter._count_crossing_objects(boxes_frame2, ids_frame2, classes_frame2, dummy_two_lines)

    assert flow_counter.merge_history == [{
        "frame": 2,
        "old_root1": "B",
        "old_root2": "A",
        "new_root": "B",
    }]

def test_get_statistics_shape_and_content(
    mocker: MockerFixture,
    dummy_two_lines: dict[str, tuple[LINE, LINE]],
    flow_counter: FlowCounter,
) -> None:
    flow_counter.model.names = {0: "car"}

    # Frame 1: "A" crosses line1 only.
    boxes_frame1 = np.array([[10, 10, 20, 20]])
    ids_frame1 = np.array(["A"])
    classes_frame1 = np.array([0])
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[True, False])
    flow_counter._count_crossing_objects(boxes_frame1, ids_frame1, classes_frame1, dummy_two_lines)

    # Frame 2: "A" and "B" merge (IoU=1.0), "B" crosses line2 and the pass
    # is counted under the merged root.
    boxes_frame2 = np.array([[10, 10, 20, 20], [10, 10, 20, 20]])
    ids_frame2 = np.array(["A", "B"])
    classes_frame2 = np.array([0, 0])
    mocker.patch("flow_counter.flow_counter.intersect", side_effect=[False, False, False, True])
    flow_counter._count_crossing_objects(boxes_frame2, ids_frame2, classes_frame2, dummy_two_lines)

    stats = flow_counter.get_statistics()

    assert stats["cls_counts"]["car"]["dummy"] == 1
    assert stats["reverse_crossings"] == {}
    # "A"'s pre-merge entry is retained alongside the merged root "B"'s,
    # since first_crossed_side records are copied to the new root rather
    # than moved.
    assert stats["first_crossed_side"] == {"dummy": {"A": "line1", "B": "line1"}}
    assert stats["merged_id_groups"] == {"B": ["A", "B"]}
    assert stats["merge_timeline"] == [
        "frame 2: object B と object A を統合（統合後のroot: B）"
    ]
    assert stats["crossing_timeline"] == [
        "frame 2: car (id=B) が dummy を通過（順方向）"
    ]

def test_get_statistics_is_empty_before_any_events(flow_counter: FlowCounter) -> None:
    stats = flow_counter.get_statistics()

    assert stats["merged_id_groups"] == {}
    assert stats["merge_timeline"] == []
    assert stats["crossing_timeline"] == []
    assert stats["first_crossed_side"] == {}
