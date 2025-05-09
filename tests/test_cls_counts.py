import numpy as np
from pytest_mock import MockerFixture

from flow_counter import FlowCounter
from flow_counter.utils import Point

def test_count_crossing_objects_updates_cls_counts(
    mocker: MockerFixture, 
    dummy_line: tuple[Point, Point], 
    flow_counter: FlowCounter, 
) -> None:
    """
    Test that cls_counts is updated correctly when an object cross the line.
    """
    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array([1])
    classes = np.array([0])

    mocker.patch("flow_counter.flow_counter.intersect", return_value=True)
    flow_counter.model.names = {0: "car"}

    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_line)

    assert flow_counter.cls_counts == {"car": 1}

def test_count_crossing_objects_accumulates_same_class(
    mocker: MockerFixture, 
    dummy_line: tuple[Point, Point], 
    flow_counter: FlowCounter, 
) -> None:
    """
    Test that cls_counts accumulates class for the same class correctly.
    """
    boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
    ids = np.array([1, 2])
    classes = np.array([0, 0])

    mocker.patch("flow_counter.flow_counter.intersect", return_value=True)
    flow_counter.model.names = {0: "car"}

    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_line)

    assert flow_counter.cls_counts == {"car": 2}

def test_count_crossing_objects_multiple_classes(
    mocker: MockerFixture, 
    dummy_line: tuple[Point, Point], 
    flow_counter: FlowCounter, 
) -> None:
    """
    Test that objects from multiple classes are counted separately.
    """
    boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
    ids = np.array([1, 2])
    classes = np.array([0, 1])

    mocker.patch("flow_counter.flow_counter.intersect", return_value=True)
    flow_counter.model.names = {0: "car", 1: "person"}

    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_line)

    assert flow_counter.cls_counts == {"car": 1, "person": 1}

def test_count_crossing_objects_skips_non_intersecting_boxes(
    mocker: MockerFixture, 
    dummy_line: tuple[Point, Point], 
    flow_counter: FlowCounter, 
) -> None:
    """
    Test that objects not intersecting the line are not counted.
    """
    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array([1])
    classes = np.array([0])

    mocker.patch("flow_counter.flow_counter.intersect", return_value=False)
    flow_counter.model.names = {0: "car"}

    result = flow_counter._count_crossing_objects(boxes, ids, classes, dummy_line)

    assert result == 0
    assert flow_counter.cls_counts == {}

def test_count_crossing_objects_skips_already_counted_id(
    mocker: MockerFixture, 
    dummy_line: tuple[Point, Point], 
    flow_counter: FlowCounter, 
) -> None:
    """
    Test that already counted object IDs are skipped.
    """
    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array([1])
    classes = np.array([0])

    mocker.patch("flow_counter.flow_counter.intersect", return_value=True)
    flow_counter.model.names = {0: "car"}
    flow_counter.counted_ids = {1}

    result = flow_counter._count_crossing_objects(boxes, ids, classes, dummy_line)

    assert result == 0
    assert flow_counter.cls_counts == {}