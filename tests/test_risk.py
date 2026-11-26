import numpy as np
import pytest

from crowdpulse.density import Density
from crowdpulse.flow import Flow, Steady
from crowdpulse.risk import ZoneTrend, level_of
from crowdpulse.zones import Reading, Zone, measure


def reading(density, inflow_per_min=0.0, area=10.0):
    return Reading(density * area, density, inflow_per_min)


def test_density_bands_follow_crowd_safety_guidance():
    assert [level_of(d) for d in (0.5, 1.99, 2.0, 3.9, 4.0, 6.0)] == ["green", "green", "amber", "amber", "red", "red"]


def test_a_steady_quiet_zone_stays_green_with_no_countdown():
    trend = ZoneTrend("Canteen", 10)
    for t in range(0, 60, 2):
        status = trend.update(t, reading(1.0))
    assert status.level == "green" and status.minutes_to_critical is None


def test_a_zone_rising_steadily_gets_a_countdown_and_turns_red_near_the_end():
    trend = ZoneTrend("Gate 6", 10)
    for t in range(0, 61):                                # 1.0 people/m2 rising by 0.5 per minute
        status = trend.update(t, reading(1.0 + 0.5 * t / 60))
    assert status.minutes_to_critical == pytest.approx(7, abs=1)     # from about 1.4 to 5 at 0.5 per minute
    assert status.level == "amber"                                   # still green by density, amber by trend
    for t in range(61, 400):
        status = trend.update(t, reading(1.0 + 0.5 * t / 60))
    assert status.level == "red" and status.minutes_to_critical <= 3


def test_people_pouring_in_raise_the_warning_before_density_rises():
    trend = ZoneTrend("Staircase", 30)
    for t in range(0, 20):                                # 1.5 people/m2, 25 people a minute crossing in
        status = trend.update(t, reading(1.5, inflow_per_min=25, area=30))
    assert status.level == "amber" and status.minutes_to_critical == pytest.approx(3.5 / (25 / 30), abs=0.5)


def test_a_brief_surge_does_not_raise_a_warning_but_a_sustained_one_does():
    trend = ZoneTrend("Balcony", 60)
    for t in range(0, 30):
        trend.update(t, reading(0.8, area=60))
    for t in range(30, 35):                                # 5 s of heavy inflow, then calm again
        status = trend.update(t, reading(0.8, inflow_per_min=200, area=60))
    assert status.level == "green"
    for t in range(35, 80):
        status = trend.update(t, reading(0.8, area=60))
    assert status.level == "green"
    for t in range(80, 100):                               # the same inflow, kept up for 20 s
        status = trend.update(t, reading(0.8, inflow_per_min=200, area=60))
    assert status.level in ("amber", "red")


def test_a_packed_zone_is_critical_now():
    trend = ZoneTrend("Balcony", 60)
    status = trend.update(0, reading(5.5, area=60))
    assert status.level == "red" and status.minutes_to_critical == 0


def converging_flow(rows=8, cols=8, cell=64, speed=40.0, inward=True):
    """Every cell moving at `speed` px/s towards (or away from) the centre of the frame."""
    centre = np.array([cols * cell / 2, rows * cell / 2])
    ys, xs = np.mgrid[0:rows, 0:cols]
    pos = np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell], axis=-1)
    direction = centre - pos if inward else pos - centre
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return Flow((direction * speed).astype(np.float32), cell)


def test_inflow_is_positive_where_a_crowd_converges_and_negative_where_it_spreads():
    density = Density(np.full((64, 64), 0.05, np.float32), scale=1.0)     # an even crowd over a 512 px frame
    centre = Zone("Centre", 10, np.array([[192, 192], [320, 192], [320, 320], [192, 320]], np.float32))
    arriving = measure([centre], density, converging_flow())["Centre"].inflow
    leaving = measure([centre], density, converging_flow(inward=False))["Centre"].inflow
    assert arriving > 0 > leaving


def test_a_steady_stream_through_the_edge_of_the_picture_neither_fills_nor_drains_a_zone():
    density = Density(np.full((64, 64), 0.05, np.float32), scale=1.0)
    down = Flow(np.tile(np.array([0.0, 60.0], np.float32), (8, 8, 1)), 64)    # everyone walking down at 60 px/s
    for name, top, bottom in (("Exit", 448, 512), ("Entry", 0, 64), ("Middle", 192, 320)):
        zone = Zone(name, 10, np.array([[0, top], [512, top], [512, bottom], [0, bottom]], np.float32))
        assert measure([zone], density, down)[name].inflow == pytest.approx(0, abs=1e-3), name


def test_people_shuffling_in_place_do_not_drain_a_crowded_zone():
    grid = np.full((64, 64), 0.01, np.float32)
    grid[24:40, 24:40] = 0.2                              # a packed square in a sparse crowd
    density = Density(grid, scale=1.0)
    square = Zone("Square", 10, np.array([[192, 192], [320, 192], [320, 320], [192, 320]], np.float32))
    jitter = np.random.default_rng(1).normal(0, 5, (8, 8, 2)).astype(np.float32)     # a few px/s, any direction
    assert measure([square], density, Flow(jitter, 64))["Square"].inflow == 0


def test_steady_keeps_persistent_motion_and_averages_jitter_away():
    rng = np.random.default_rng(2)
    steady = Steady(seconds=4)
    for t in range(40):
        walking = np.tile(np.array([30.0, 0.0], np.float32), (4, 4, 1))
        out = steady.update(float(t), Flow(walking + rng.normal(0, 25, (4, 4, 2)).astype(np.float32), 64))
    assert np.allclose(out.vectors.mean(axis=(0, 1)), [30, 0], atol=8)
    assert out.vectors[..., 0].std() < 15                # jitter of 25 px/s shrinks well below that


def test_zone_head_count_is_the_density_inside_its_polygon():
    grid = np.zeros((64, 64), np.float32)
    grid[10:20, 10:20] = 0.1                              # 10 people in an 80x80 px square at (80..160)
    density = Density(grid, scale=1.0)
    zone = Zone("Square", 4, np.array([[80, 80], [160, 80], [160, 160], [80, 160]], np.float32))
    still = Flow(np.zeros((8, 8, 2), np.float32), 64)
    got = measure([zone], density, still)["Square"]
    assert got.people == pytest.approx(10, abs=0.01) and got.density == pytest.approx(2.5, abs=0.01)


def test_a_few_seconds_of_rising_readings_at_the_start_are_not_a_trend():
    from crowdpulse.risk import ZoneTrend
    from crowdpulse.zones import Reading
    trend = ZoneTrend("Staircase", 30)
    for second in range(12):                                  # 0.84 to 0.93 people per m2: about 3 people of noise
        density = 0.84 + 0.0082 * second
        status = trend.update(float(second), Reading(density * 30, density, 0.0))
    assert status.level == "green" and status.rate == 0.0
