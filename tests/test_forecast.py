import numpy as np

from crowdpulse.flow import Flow
from crowdpulse.forecast import forecast, zone_ahead
from crowdpulse.zones import Zone

CELL = 10


def corridor(people_at: dict[int, float], walking: float) -> tuple[Flow, np.ndarray]:
    """One row of 10 cells; people in cells 0-5 walk at `walking` px/s, anyone in cells 6-9 stands still."""
    people = np.zeros((1, 10), np.float32)
    for col, n in people_at.items():
        people[0, col] = n
    vectors = np.zeros((1, 10, 2), np.float32)
    vectors[0, :6, 0] = walking
    return Flow(vectors, CELL), people


def test_people_walking_into_a_standing_crowd_collect_there():
    flow, people = corridor({1: 3, 3: 2, 8: 10}, walking=20)
    ahead = forecast(flow, people)
    assert ahead.grid[0, 8] == 15                                # both groups cross the empty cells and stop at the crowd
    assert ahead.grid.sum() == 15                                # nobody is lost


def test_people_walking_out_of_the_picture_are_dropped():
    flow, people = corridor({2: 4, 8: 1}, walking=-20)
    assert forecast(flow, people).grid.sum() == 1


def test_a_zone_in_front_of_a_stream_fills_up_in_the_forecast():
    flow, people = corridor({1: 3, 3: 2, 8: 10}, walking=20)
    zone = Zone("End", 4.0, np.array([[60, 0], [100, 0], [100, 10], [60, 10]], np.float32))   # cells 6-9
    [(name, density)] = zone_ahead([zone], people, forecast(flow, people), {"End": 2.5}).items()
    assert name == "End" and density == 2.5 + 5 / 4.0


def test_people_walking_through_an_empty_doorway_carry_on_out_of_the_picture():
    flow, people = corridor({1: 3}, walking=20)                  # nobody standing in cells 6-9
    assert forecast(flow, people).grid.sum() == 0
