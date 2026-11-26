from crowdpulse.alerts import AlertLog
from crowdpulse.risk import Status


def status(level, density=1.0, inflow=0.0, minutes=None, name="Gate 6"):
    return Status(name, density * 30, density, inflow, 0.0, minutes, level)


def test_an_alert_is_raised_each_time_a_zone_gets_worse_and_once_when_it_settles():
    log = AlertLog()
    seen = []
    levels = ["green", "green", "amber", "amber", "red", "red", "amber", "green", "green", "green", "green"]
    for t, level in zip(range(0, 200, 10), levels):
        seen += [(a.t, a.level) for a in log.update(t, {"Gate 6": status(level)})]
    assert seen == [(20, "amber"), (40, "red"), (100, "green")]   # stepping down to amber stays quiet; green after 30 s


def test_a_zone_flickering_between_amber_and_red_alerts_once():
    log = AlertLog()
    seen = []
    levels = ["amber", "red", "amber", "red", "green", "amber", "red", "green", "green", "green", "green", "red"]
    for t, level in zip(range(0, 200, 10), levels):
        seen += [(a.t, a.level) for a in log.update(t, {"Gate 6": status(level)})]
    assert seen == [(0, "amber"), (10, "red"), (100, "green"), (110, "red")]   # a short green dip is not "back to normal"


def test_alerts_say_what_is_happening_and_what_to_do():
    log = AlertLog()
    [alert] = log.update(10, {"Gate 6": status("amber", density=2.4, inflow=30, minutes=6)})
    assert "Gate 6 is crowded: 2.4 people per m², 30 people a minute walking in" in alert.message
    assert "about 6 minutes" in alert.message and "hold people at its entrances" in alert.action
    [alert] = log.update(20, {"Gate 6": status("red", density=4.6, minutes=0)})
    assert "dangerously crowded" in alert.message and "crush-risk level now" in alert.message
    assert alert.action.startswith("Stop people entering Gate 6 now")


def test_an_early_warning_says_where_the_zone_is_heading_not_that_it_is_already_dangerous():
    log = AlertLog()
    [amber] = log.update(0, {"Gate 6": status("amber", density=0.6, inflow=90, minutes=6)})
    [red] = log.update(60, {"Gate 6": status("red", density=1.7, inflow=167, minutes=2)})
    assert amber.message.startswith("Gate 6 is filling fast: 0.6 people per m²")
    assert red.message.startswith("Gate 6 is heading for a crush: 1.7 people per m²") and "about 2 minutes" in red.message
