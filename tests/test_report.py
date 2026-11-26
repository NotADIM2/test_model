from crowdpulse.report import build, zone_summary


def record(t, density, level, inflow=0.0, name="Gate 3"):
    return {"t": t, "people": density * 96, "zones": [{"name": name, "people": density * 96, "density": density,
                                                       "inflow_per_min": inflow, "level": level, "minutes_to_critical": None}]}


def test_zone_summary_finds_the_peak_and_time_at_each_level():
    history = [record(0, 1.0, "green"), record(10, 2.5, "amber", 40), record(20, 4.6, "red", 90), record(30, 3.0, "amber")]
    [zone] = zone_summary(history)
    assert (zone["peak"], zone["peak_at"], zone["peak_inflow"]) == (4.6, 20, 90)
    assert zone["seconds"] == {"green": 0.0, "amber": 20.0, "red": 10.0}


def test_report_states_the_outcome_and_escapes_names_from_files():
    history = [record(0, 1.0, "green", name="<b>Gate</b>"), record(10, 4.6, "red", name="<b>Gate</b>")]
    alert = {"t": 10, "zone": "<b>Gate</b>", "level": "red", "message": "m", "action": "a"}
    page = build("canteen.mp4", history, [alert], [{"t": 12, "what": "Opened Gate 2"}], None, None)
    assert "A zone reached red" in page and "Opened Gate 2" in page
    assert "<b>Gate</b>" not in page and "&lt;b&gt;Gate&lt;/b&gt;" in page
    quiet = build("canteen.mp4", [record(0, 1.0, "green")], [], [], None, None)
    assert "No crowding warnings during this run." in quiet
