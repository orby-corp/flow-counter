import numpy as np
import pytest
from pytest_mock import MockerFixture

from flow_counter import FlowCounter
from flow_counter.utils import Point

@pytest.fixture
def dummy_line() -> tuple[Point, Point]:
    return ((0, 0), (100, 100))

@pytest.fixture
def flow_counter(mocker: MockerFixture) -> FlowCounter:
    """
    Returns a FlowCounter instance with a mocked YOLO model to avoid real model loading.
    """
    mock_yolo = mocker.patch("flow_counter.flow_counter.YOLO", autospec=True)
    return FlowCounter("dummy_model.pt")

def test_count_crossing_objects_counts_new_ids(
    mocker: MockerFixture, 
    dummy_line: tuple[Point, Point], 
    flow_counter: FlowCounter, 
) -> None:
    """
    Test that a new object with a valid ID that intersects the line is counted.
    """
    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array([1])

    mocker.patch("flow_counter.flow_counter.intersect", return_value=True)

    result = flow_counter._count_crossing_objects(boxes, ids, dummy_line)

    assert result == 1
    assert 1 in flow_counter.counted_ids

def test_count_crossing_objects_skips_non_intersection(
    mocker: MockerFixture, 
    dummy_line: tuple[Point, Point], 
    flow_counter: FlowCounter, 
) -> None:
    """
    Test that an object which does not intersect the line is not counted.
    """
    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array([2])

    mocker.patch("flow_counter.flow_counter.intersect", return_value=False)

    result = flow_counter._count_crossing_objects(boxes, ids, dummy_line)

    assert result == 0
    assert 2 not in flow_counter.counted_ids

def test_count_crossing_objects_skips_already_counted(
    mocker: MockerFixture, 
    dummy_line: tuple[Point, Point], 
    flow_counter: FlowCounter, 
) -> None:
    """
    Test that an object ID that has already been counted is skipped.
    """
    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array([3])
    flow_counter.counted_ids.add(3)

    mocker.patch("flow_counter.flow_counter.intersect", return_value=True)

    result = flow_counter._count_crossing_objects(boxes, ids, dummy_line)

    assert result == 0
    assert flow_counter.counted_ids == {3}

def test_count_crossing_objects_skips_invalid_id(
    mocker: MockerFixture, 
    dummy_line: tuple[Point, Point], 
    flow_counter: FlowCounter, 
) -> None:
    """
    Test that an object with ID -1 (invalid tracking ID) is skipped.
    """
    boxes = np.array([[10, 10, 20, 20]])
    ids = np.array([-1])

    mocker.patch("flow_counter.flow_counter.intersect", return_value=True)

    result = flow_counter._count_crossing_objects(boxes, ids, dummy_line)

    assert result == 0
    assert flow_counter.counted_ids == set()