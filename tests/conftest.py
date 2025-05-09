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