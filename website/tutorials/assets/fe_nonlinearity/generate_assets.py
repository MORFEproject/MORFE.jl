#!/usr/bin/env python3
"""Build the assets of the "Writing a finite element nonlinearity" tutorial.

Nothing is measured here. The costs are read from two committed copies of CSVs that
MORFEExamples' `writing_a_finite_element_nonlinearity` example writes:

- `cost.v2.csv` is `results/data/rhs_cost.csv`, written by the notebook: the tutorial's own
  first-Piola kernel, as a closure and through the element-level interface with flux
  summation;
- `benchmark.v2.csv` is the output of `benchmark.jl` for MORFEFerrite's production SVK kernel
  behind the same two routes, one row per assembled monomial, on the generated mesh with the
  tutorial's 4977 dofs:

      MORFE_BENCH_MESHES=small MORFE_BENCH_ROMS=2 MORFE_BENCH_ORDERS="2=>9" \
      MORFE_BENCH_OUT=benchmark.v2.csv julia benchmark.jl

  The costs up to each order are the sums of those rows; the solve is graded, so one run to
  order 9 contains every lower order.

To refresh, rerun the example, copy both CSVs here under these names and run, from the
MORFE repository root:

    python3 website/tutorials/assets/fe_nonlinearity/generate_assets.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOSURE, ELEMENT = "#cb3c33", "#4063d8"   # Julia red and blue


def read_csv(path: Path):
    with path.open() as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    header, body = rows[0], rows[1:]
    return [dict(zip(header, r)) for r in body]


def up_to(rows, route, order):
    """The benchmark rows of one route whose monomials have degree at most `order`."""
    return [r for r in rows if r["route"] == route and int(r["degree"]) <= order]


def load():
    tutorial = [
        dict(order=int(r["order"]), passes=(int(r["passes_closure"]), int(r["passes_element"])),
             time=(float(r["rhs_closure_s"]), float(r["rhs_element_s"])))
        for r in read_csv(HERE / "cost.v2.csv")
    ]
    bench = read_csv(HERE / "benchmark.v2.csv")
    production = [
        dict(order=p,
             passes=tuple(sum(int(r["passes"]) for r in up_to(bench, route, p))
                          for route in ("closure", "single_pass")),
             time=tuple(sum(float(r["rhs_time_s"]) for r in up_to(bench, route, p))
                        for route in ("closure", "single_pass")))
        for p in (r["order"] for r in tutorial)
    ]
    # Both runs count the same traversals: they depend on the multiindex set, not the kernel.
    assert [r["passes"] for r in tutorial] == [r["passes"] for r in production]
    return tutorial, production


# ── chart geometry (viewBox 900 x 430) ─────────────────────────────────────────────────
W, H = 900, 430
TOP, BOTTOM = 62, 368
LEFT_PANEL = (78, 396)
RIGHT_PANEL = (548, 852)
ORDERS = (3, 5, 7, 9)
MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"


def xmap(panel, order):
    x0, x1 = panel
    return x0 + (order - 2.6) / (9.4 - 2.6) * (x1 - x0)


def ylog(v, lo=3.0, hi=700.0):
    return BOTTOM - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (BOTTOM - TOP)


def ylin(v, hi):
    return BOTTOM - v / hi * (BOTTOM - TOP)


def nice_axis(v):
    """Upper limit and tick step with round ticks and some head-room above `v`."""
    for step in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5):
        hi = step * math.ceil(1.08 * v / step)
        if hi / step <= 6:
            return hi, step
    return math.ceil(v), 1


def frame(panel, yticks, ylabel):
    x0, x1 = panel
    out = []
    for y, label in yticks:
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 9}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
    for o in ORDERS:
        x = xmap(panel, o)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{TOP}" y2="{BOTTOM}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{BOTTOM + 20}" text-anchor="middle">{o}</text>')
    out.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{BOTTOM}" y2="{BOTTOM}"/>')
    out.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="{TOP}" y2="{BOTTOM}"/>')
    out.append(f'<text class="label" x="{(x0 + x1) / 2:.1f}" y="{BOTTOM + 46}" text-anchor="middle">expansion order</text>')
    out.append(f'<text class="label" transform="translate({x0 - 52} {(TOP + BOTTOM) / 2:.1f}) rotate(-90)" '
               f'text-anchor="middle">{ylabel}</text>')
    return out


def series(panel, rows, key, ymapf, unit):
    out = []
    for route, color, name in ((0, CLOSURE, "closure"), (1, ELEMENT, "element-level")):
        pts = [(xmap(panel, r["order"]), ymapf(r[key][route]), r) for r in rows]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{path}"/>')
        for x, y, r in pts:
            value = r[key][route]
            text = f"{value:g}" if key == "passes" else f"{value:.2f} s"
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" stroke="#07070b" stroke-width="1.2">'
                       f'<title>order {r["order"]} · {name} · {text}{unit}</title></circle>')
    return out


def time_group(rows, gid, visible):
    hi, step = nice_axis(max(max(r["time"]) for r in rows))
    n = round(hi / step)
    ticks = [(ylin(step * k, hi), f"{round(step * k, 6):g}") for k in range(n + 1)]
    out = [f'<g id="{gid}" style="display:{"inline" if visible else "none"}">']
    out += frame(RIGHT_PANEL, ticks, "right-hand-side time [s]")
    out += series(RIGHT_PANEL, rows, "time", lambda v: ylin(v, hi), "")
    last = rows[-1]
    ratio = last["time"][0] / last["time"][1]
    x = xmap(RIGHT_PANEL, last["order"]) + 12
    y = (ylin(last["time"][0], hi) + ylin(last["time"][1], hi)) / 2
    out.append(f'<text class="ratio" x="{x:.1f}" y="{y + 4:.1f}" text-anchor="start">×{ratio:.1f}</text>')
    out.append("</g>")
    return out


def chart_svg(tutorial, production):
    passes_ticks = [(ylog(v), str(v)) for v in (3, 10, 30, 100, 300)]
    body = frame(LEFT_PANEL, passes_ticks, "mesh traversals")
    body += series(LEFT_PANEL, tutorial, "passes", ylog, " traversals")
    last = tutorial[-1]
    x = xmap(LEFT_PANEL, last["order"]) + 12
    body.append(f'<text class="ratio" x="{x:.1f}" y="{(ylog(last["passes"][0]) + ylog(last["passes"][1])) / 2 + 4:.1f}" '
                f'text-anchor="start">×{last["passes"][0] / last["passes"][1]:.0f}</text>')
    body += time_group(tutorial, "time-tutorial", True)
    body += time_group(production, "time-production", False)
    legend = (
        f'<g transform="translate({W / 2 - 150} 24)">'
        f'<line x1="0" x2="22" y1="0" y2="0" stroke="{CLOSURE}" stroke-width="2.2"/><circle cx="11" cy="0" r="4" fill="{CLOSURE}"/>'
        f'<text class="label" x="30" y="4">closure</text>'
        f'<line x1="140" x2="162" y1="0" y2="0" stroke="{ELEMENT}" stroke-width="2.2"/><circle cx="151" cy="0" r="4" fill="{ELEMENT}"/>'
        f'<text class="label" x="170" y="4">element-level interface</text></g>'
    )
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mesh traversals and right-hand-side time against '
            f'the expansion order, for the closure and the element-level interface">' + legend + "".join(body) + "</svg>")


COST_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Closure against element-level interface</title>
<style>
:root{--bg:#0a0a0f;--stage:#07070b;--ink:#e8e8ee;--ink2:#a0a0b0;--ink3:#6e6e7e;--hair:#26262f;--purple:#9558b2}*{box-sizing:border-box}html,body{margin:0;height:100%}body{background:var(--bg);color:var(--ink);overflow:hidden;font:13px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif}
#wrap{display:flex;flex-direction:column;height:100%;padding:10px 12px;gap:8px}#bar{display:flex;gap:6px;align-items:center;min-height:28px}button{font:inherit;font-size:12px;padding:4px 10px;border-radius:5px;cursor:pointer;background:transparent;color:var(--ink2);border:1px solid var(--hair)}button:hover{border-color:var(--purple);color:var(--ink)}button.on{background:rgba(149,88,178,.16);border-color:var(--purple);color:var(--ink)}#meta{margin-left:auto;color:var(--ink3);font-size:11px;font-family:__MONO__;letter-spacing:.04em}
#stage{flex:1;min-height:0;border:1px solid var(--hair);border-radius:6px;background:var(--stage);overflow:hidden}svg{display:block;width:100%;height:100%}.axis{stroke:var(--ink3);stroke-width:1}.grid{stroke:var(--hair);stroke-width:.8}.tick,.label{fill:var(--ink3);font:11px __MONO__}.label{fill:var(--ink2)}.ratio{fill:var(--ink2);font:600 12px __MONO__}circle{cursor:default}
#note{color:var(--ink3);font-size:11.5px;min-height:1.3em}
</style></head><body><div id="wrap"><div id="bar"><button id="tutorial" class="on">tutorial kernel</button><button id="production">MORFEFerrite kernel</button><span id="meta">clamped beam · 4977 dofs · master mode 1</span></div>
<div id="stage">__SVG__</div>
<div id="note">The traversals do not depend on the kernel. Hover a point for its value.</div></div>
<script>
const notes={tutorial:'The notebook’s first-Piola kernel. The traversals do not depend on the kernel. Hover a point for its value.',production:'MORFEFerrite’s built-in SVK kernel, timed by benchmark.jl. It forms one stress per distinct coefficient rather than one per factorisation, so its single pass gains even more.'};
function show(k){for(const id of['tutorial','production']){document.getElementById(id).classList.toggle('on',id===k);document.getElementById('time-'+id).style.display=id===k?'inline':'none'}document.getElementById('note').textContent=notes[k]}
document.getElementById('tutorial').onclick=()=>show('tutorial');document.getElementById('production').onclick=()=>show('production');show('tutorial');
</script></body></html>'''


def thumbnail_svg(tutorial):
    panel = (70, 360)
    lo, hi = 3.0, 700.0
    top, bottom = 34, 186

    def x(o):
        return panel[0] + (o - 3) / 6 * (panel[1] - panel[0])

    def y(v):
        return bottom - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (bottom - top)

    parts = ['<rect width="400" height="225" fill="#0a0a0f"/>']
    for v in (10, 100):
        parts.append(f'<line x1="{panel[0]}" x2="{panel[1]}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#26262f" stroke-width="1"/>')
    parts.append(f'<line x1="{panel[0]}" x2="{panel[1]}" y1="{bottom}" y2="{bottom}" stroke="#4a4a58" stroke-width="1.2"/>')
    parts.append(f'<line x1="{panel[0]}" x2="{panel[0]}" y1="{top - 6}" y2="{bottom}" stroke="#4a4a58" stroke-width="1.2"/>')
    for route, color, label in ((0, CLOSURE, "closure"), (1, ELEMENT, "element-level")):
        pts = [(x(r["order"]), y(r["passes"][route])) for r in tutorial]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="'
                     + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
        for a, b in pts:
            parts.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4.5" fill="{color}"/>')
        a, b = pts[-1]
        n = tutorial[-1]["passes"][route]
        parts.append(f'<text x="{a - 8:.1f}" y="{b + (-10 if route == 0 else 40):.1f}" text-anchor="end" '
                     f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="13" fill="{color}">{label} · {n}</text>')
    parts.append(f'<text x="{panel[0]}" y="{bottom + 26}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11" fill="#6e6e7e">'
                 'mesh traversals, orders 3 to 9</text>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 225" role="img" aria-labelledby="title desc">'
            '<title id="title">Mesh traversals of a closure and of the element-level interface</title>'
            '<desc id="desc">Two lines against the expansion order on a log scale: the closure rises to 486 '
            'traversals, the element-level interface to 28.</desc>' + "".join(parts) + "</svg>")


def main():
    tutorial, production = load()
    (HERE / "cost.html").write_text(
        COST_HTML.replace("__SVG__", chart_svg(tutorial, production)).replace("__MONO__", MONO))
    (HERE / "thumb.svg").write_text(thumbnail_svg(tutorial))
    print("wrote cost.html and thumb.svg")


if __name__ == "__main__":
    main()
