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

def test_count_crossing_objects_only_vehicles(
    mocker, dummy_line, flow_counter
):
    """
    Only vehicle classes in counted_cls_name should be counted.
    """
    boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
    ids = np.array([1, 2])
    classes = np.array([0, 1])  # 0: "car", 1: "dog"

    mocker.patch("flow_counter.flow_counter.intersect", return_value=True)
    flow_counter.model.names = {0: "car", 1: "dog"}

    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_line)

    # Only 'car' should be counted
    assert flow_counter.cls_counts == {"car": 1}

def test_count_crossing_objects_multiple_vehicle_types(
    mocker, dummy_line, flow_counter
):
    """
    Different vehicle types are counted separately if they cross.
    """
    boxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]])
    ids = np.array([1, 2, 3])
    classes = np.array([0, 1, 2])  # 0: car, 1: bus, 2: dog

    mocker.patch("flow_counter.flow_counter.intersect", return_value=True)
    flow_counter.model.names = {0: "car", 1: "bus", 2: "dog"}

    flow_counter._count_crossing_objects(boxes, ids, classes, dummy_line)

    # Only vehicles (car, bus) should be counted, dog should not
    assert flow_counter.cls_counts == {"car": 1, "bus": 1}

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