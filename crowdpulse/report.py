"""The incident report: one self-contained HTML file describing a run, to save, print or hand over.

It holds everything inline (styles, the chart as SVG, the worst frame as a JPEG), so it opens offline on any
computer and nothing in it is fetched from the internet.
"""
import base64
from datetime import datetime
from html import escape

from .risk import AMBER_FROM, RED_FROM

LINE_COLOURS = ("#1d4ed8", "#7c3aed", "#0e7490", "#be185d", "#4d7c0f", "#c2410c")
LEVEL_WORDS = {"green": "Normal", "amber": "Amber", "red": "Red"}


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def zone_summary(history: list[dict]) -> list[dict]:
    """Per zone: peak density and when, peak inflow, and seconds spent at each level."""
    zones: dict[str, dict] = {}
    for before, record in zip([None] + history[:-1], history):
        step = record["t"] - before["t"] if before else 0.0
        for z in record["zones"]:
            s = zones.setdefault(z["name"], {"name": z["name"], "peak": 0.0, "peak_at": record["t"], "peak_inflow": 0.0,
                                             "seconds": {"green": 0.0, "amber": 0.0, "red": 0.0}})
            if z["density"] > s["peak"]:
                s["peak"], s["peak_at"] = z["density"], record["t"]
            s["peak_inflow"] = max(s["peak_inflow"], z["inflow_per_min"])
            s["seconds"][z["level"]] += step
    return list(zones.values())


def chart_svg(history: list[dict], width: int = 900, height: int = 260) -> str:
    names = list(dict.fromkeys(z["name"] for r in history for z in r["zones"]))
    if len(history) < 2 or not names:
        return "<p class='muted'>Not enough readings for a chart.</p>"
    left, right, top, bottom = 40, 10, 10, 28
    w, h = width - left - right, height - top - bottom
    peak = max(6.0, *(z["density"] for r in history for z in r["zones"]))
    t0, t1 = history[0]["t"], history[-1]["t"]
    x = lambda t: left + (t - t0) / max(t1 - t0, 1e-6) * w
    y = lambda d: top + h - d / peak * h
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Density by zone over time'>"]
    for low, high, colour in ((0, AMBER_FROM, "#e6f4ea"), (AMBER_FROM, RED_FROM, "#fef3c7"), (RED_FROM, peak, "#fde8e8")):
        parts.append(f"<rect x='{left}' y='{y(high):.1f}' width='{w}' height='{y(low) - y(high):.1f}' fill='{colour}'/>")
    for d in range(0, int(peak) + 1, 2):
        parts.append(f"<text x='{left - 6}' y='{y(d) + 4:.1f}' text-anchor='end'>{d}</text>")
    for t in (t0, (t0 + t1) / 2, t1):
        parts.append(f"<text x='{x(t):.1f}' y='{height - 8}' text-anchor='middle'>{clock(t)}</text>")
    for i, name in enumerate(names):
        points = " ".join(f"{x(r['t']):.1f},{y(z['density']):.1f}" for r in history for z in r["zones"] if z["name"] == name)
        parts.append(f"<polyline points='{points}' fill='none' stroke='{LINE_COLOURS[i % len(LINE_COLOURS)]}' stroke-width='2'/>")
    parts.append("</svg>")
    legend = "".join(f"<span style='--c:{LINE_COLOURS[i % len(LINE_COLOURS)]}'>{escape(n)}</span>" for i, n in enumerate(names))
    return "".join(parts) + f"<div class='legend'>{legend}</div>"


def build(source: str, history: list[dict], alerts: list[dict], actions: list[dict],
          worst_frame: bytes | None, worst_record: dict | None, made: datetime | None = None) -> str:
    made = made or datetime.now()
    zones = zone_summary(history)
    duration = history[-1]["t"] - history[0]["t"] if len(history) > 1 else 0.0
    reds = [a for a in alerts if a["level"] == "red"]
    first_warning = next((a for a in alerts if a["level"] != "green"), None)
    worst_level = "red" if reds else "amber" if first_warning else "green"
    headline = {"red": "A zone reached red: dangerous crowding, or a crush within minutes at the rate it was filling.",
                "amber": "Crowding built up but no zone reached red.",
                "green": "No crowding warnings during this run."}[worst_level]

    facts = [("Source", source), ("Covered", f"{clock(history[0]['t'])} to {clock(history[-1]['t'])} ({clock(duration)} long)" if history else "nothing"),
             ("Most people in view", f"{max((r['people'] for r in history), default=0):.0f}"),
             ("First warning", f"{clock(first_warning['t'])}, {escape(first_warning['zone'])} ({first_warning['level']})" if first_warning else "none"),
             ("Red alerts", str(len(reds)))]
    fact_rows = "".join(f"<dt>{k}</dt><dd>{escape(v) if k == 'Source' else v}</dd>" for k, v in facts)

    zone_rows = "".join(
        f"<tr><td>{escape(z['name'])}</td><td class='num'>{z['peak']:.1f}</td><td class='num'>{clock(z['peak_at'])}</td>"
        f"<td class='num'>{z['peak_inflow']:.0f}</td>"
        + "".join(f"<td class='num {lvl}'>{clock(z['seconds'][lvl])}</td>" for lvl in ("green", "amber", "red")) + "</tr>"
        for z in zones) or "<tr><td colspan='7' class='muted'>No zones were set for this run.</td></tr>"

    alert_rows = "".join(
        f"<tr class='{a['level']}'><td class='num'>{clock(a['t'])}</td><td>{escape(a['zone'])}</td>"
        f"<td><span class='pill {a['level']}'>{LEVEL_WORDS[a['level']]}</span></td>"
        f"<td>{escape(a['message'])}<br><b>{escape(a['action'])}</b></td></tr>" for a in alerts) \
        or "<tr><td colspan='4' class='muted'>No alerts.</td></tr>"

    action_block = ""
    if actions:
        rows = "".join(f"<tr><td class='num'>{clock(a['t'])}</td><td>{escape(a['what'])}</td></tr>" for a in actions)
        action_block = f"<h2>What the operator did</h2><table><thead><tr><th>Time</th><th>Action</th></tr></thead><tbody>{rows}</tbody></table>"

    frame_block = ""
    if worst_frame and worst_record:
        top = max(worst_record["zones"], key=lambda z: z["density"], default=None)
        caption = f"At {clock(worst_record['t'])}" + (f": {escape(top['name'])} at {top['density']:.1f} people per m²" if top else "")
        frame_block = (f"<h2>The worst moment</h2><figure><img alt='Annotated view at the worst moment' "
                       f"src='data:image/jpeg;base64,{base64.b64encode(worst_frame).decode()}'><figcaption>{caption}</figcaption></figure>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CrowdPulse incident report</title>
<style>
body {{ font-family: "Segoe UI", system-ui, sans-serif; color: #111827; background: #fff; margin: 0; font-size: 14px; }}
main {{ max-width: 960px; margin: 0 auto; padding: 32px 24px 48px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 17px; margin: 32px 0 10px; }}
.muted {{ color: #5b6475; }}
.headline {{ font-size: 16px; font-weight: 600; padding: 12px 16px; border-radius: 8px; margin: 20px 0; }}
.headline.green {{ background: #e6f4ea; color: #16833f; }} .headline.amber {{ background: #fef3c7; color: #b45309; }}
.headline.red {{ background: #fde8e8; color: #c62828; }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 20px; margin: 0; }} dt {{ color: #5b6475; }} dd {{ margin: 0; font-weight: 600; }}
table {{ width: 100%; border-collapse: collapse; }} th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2e5ea; vertical-align: top; }}
th {{ font-size: 12px; color: #5b6475; font-weight: 600; }} .num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
td.amber {{ color: #b45309; }} td.red {{ color: #c62828; font-weight: 600; }}
.pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; white-space: nowrap; }}
.pill.green {{ background: #e6f4ea; color: #16833f; }} .pill.amber {{ background: #fef3c7; color: #b45309; }} .pill.red {{ background: #fde8e8; color: #c62828; }}
svg {{ width: 100%; height: auto; font-size: 12px; fill: #5b6475; }}
.legend {{ display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: #5b6475; margin-top: 6px; }}
.legend span::before {{ content: ""; display: inline-block; width: 10px; height: 3px; background: var(--c); margin-right: 6px; vertical-align: middle; }}
figure {{ margin: 0; }} figure img {{ width: 100%; border-radius: 8px; }} figcaption {{ color: #5b6475; margin-top: 6px; }}
.note {{ margin-top: 40px; font-size: 12px; color: #5b6475; }}
@media print {{ main {{ padding: 0; }} h2 {{ break-after: avoid; }} tr, figure {{ break-inside: avoid; }} }}
</style></head>
<body><main>
<h1>CrowdPulse incident report</h1>
<div class="muted">Made {made:%d %B %Y, %H:%M} on this laptop</div>
<div class="headline {worst_level}">{headline}</div>
<dl>{fact_rows}</dl>
<h2>Zones</h2>
<table><thead><tr><th>Zone</th><th>Peak people/m²</th><th>Peak at</th><th>Most walking in a minute</th>
<th>Time green</th><th>Time amber</th><th>Time red</th></tr></thead><tbody>{zone_rows}</tbody></table>
<h2>Density over time <span class="muted">people per m²; amber from {AMBER_FROM}, red from {RED_FROM}</span></h2>
{chart_svg(history)}
{frame_block}
<h2>Alerts</h2>
<table><thead><tr><th>Time</th><th>Zone</th><th>Level</th><th>What happened and what to do</th></tr></thead><tbody>{alert_rows}</tbody></table>
{action_block}
<p class="note">Densities are estimates from the camera (or the simulation) and the zone areas entered for it. Treat the report as a
record of what CrowdPulse showed, alongside what stewards saw on the ground.</p>
</main></body></html>
"""
