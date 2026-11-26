from crowdpulse.alerts import AlertLog
from crowdpulse.risk import Status


def status(level, density=1.0, inflow=0.0, minutes=None, name="Gate 6"):
    return Status(name, density * 30, density, inflow, 0.0, minutes, level)


def test_an_alert_is_raised_each_time_a_zone_gets_worse_and_once_when_it_settles():
    log = AlertLog()
    seen = []
    for t, level in enumerate(["green", "green", "amber", "amber", "red", "red", "amber", "green", "green"]):
        seen += [(a.t, a.level) for a in log.update(t, {"Gate 6": status(level)})]
    assert seen == [(2, "amber"), (4, "red"), (7, "green")]       # stepping down from red to amber stays quiet


def test_alerts_say_what_is_happening_and_what_to_do():
    log = AlertLog()
    [alert] = log.update(10, {"Gate 6": status("amber", density=2.4, inflow=30, minutes=6)})
    assert "Gate 6 is filling up: 2.4 people per m², 30 people a minute walking in" in alert.message
    assert "about 6 minutes" in alert.message and "hold people at its entrances" in alert.action
    [alert] = log.update(20, {"Gate 6": status("red", density=4.6, minutes=0)})
    assert "dangerously crowded" in alert.message and "crush-risk level now" in alert.message
    assert alert.action.startswith("Stop people entering Gate 6 now")
