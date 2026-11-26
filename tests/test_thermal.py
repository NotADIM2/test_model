import numpy as np

from crowdpulse.sim import PX, TOP, Simulation
from crowdpulse.thermal import ThermalModel


def plaza(people: np.ndarray) -> Simulation:
    sim = Simulation()
    sim.pos = people.astype(np.float32)
    return sim


def test_an_empty_cool_scene_has_nobody_in_it():
    model = ThermalModel()
    assert model.predict(plaza(np.zeros((0, 2))).render_thermal()).count == 0
    assert model.person_px is None                             # nothing to measure a person by yet


def test_counts_scattered_people_from_their_heat():
    rng = np.random.default_rng(1)
    people = np.stack([rng.uniform(2, 78, 120), rng.uniform(2, 48, 120)], axis=1)
    count = ThermalModel().predict(plaza(people).render_thermal()).count
    assert abs(count - 120) <= 0.15 * 120


def test_a_packed_crowd_still_reads_as_dangerous_after_calibrating_on_a_sparse_one():
    rng = np.random.default_rng(2)
    model = ThermalModel()
    sparse = np.stack([rng.uniform(2, 78, 60), rng.uniform(15, 48, 60)], axis=1)
    model.predict(plaza(sparse).render_thermal())                # calibrates on separate people
    packed = np.stack([rng.uniform(20, 30, 500), rng.uniform(0.5, 10.5, 500)], axis=1)   # 5 people per m2
    density = model.predict(plaza(packed).render_thermal())
    sim = plaza(packed)
    zone = sim.pixel_zones()[1]                                  # Gate 2: 12 m x 8 m centred on x = 30
    zone_area = zone.area_m2
    inside = (density.grid * zone.mask(density.grid.shape, 8 / density.scale)).sum()
    true_inside = zone.contains(sim.pos * PX + [0, TOP]).sum()
    assert inside / zone_area >= 0.7 * true_inside / zone_area   # undercounts a packed crowd, but not by much
    assert inside / zone_area >= 2.0                             # and it still reads as crowded
