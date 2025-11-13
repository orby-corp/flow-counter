import pytest
from pytest_mock import MockerFixture

from flow_counter import FlowCounter
from flow_counter.utils import Point

LINE = tuple[Point, Point]

@pytest.fixture
def dummy_line() -> dict[str, tuple[LINE, LINE]]:
    return {"dummy": (((0, 0), (100, 100)), ((0, 0), (100, 100)))}

@pytest.fixture
def dummy_two_lines() -> dict[str, tuple[LINE, LINE]]:
    return {"dummy": (((0, 0), (100, 0)), ((0, 100), (100, 100)))}

@pytest.fixture
def flow_counter(mocker: MockerFixture) -> FlowCounter:
    """
    Returns a FlowCounter instance with a mocked YOLO model to avoid real model loading.
    """
    mock_yolo = mocker.patch("flow_counter.flow_counter.YOLO", autospec=True)
    return FlowCounter("dummy_model.pt")