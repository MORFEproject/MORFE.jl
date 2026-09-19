#!/usr/bin/env python3
"""Render mems_micromirror.html's code blocks from the notebook's code cells.

The tutorial page quotes MORFEExamples'
`dual_axis_mems_micromirror/dual_axis_mems_micromirror.ipynb`. As on the Kármán page, every
`<pre class="code">` carries a `<!-- nbcode:N -->` marker naming the notebook code cell it
shows, and this rewrites each one from that cell, syntax-highlighted, with the
documentation links reapplied, so the page and the notebook cannot drift apart.

    python3 website/tutorials/assets/mems_micromirror/sync_code_blocks.py [--check]

`--check` exits nonzero if the page is out of date instead of rewriting it. The notebook's
cell count and the page's marker set must agree, so a new cell fails loudly rather than
being dropped.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE.parents[1] / "mems_micromirror.html"
REPO = HERE.parents[3]          # …/MORFE_jl
EXAMPLES = next(p for p in (REPO.parent / "MORFEExamples",
                            REPO.parent / "MORFEExamples" / "MORFEExamples") if p.is_dir())
NB = EXAMPLES / "dual_axis_mems_micromirror" / "dual_axis_mems_micromirror.ipynb"

DOCS = "../documentation.html#"
FDOCS = "../morfeferrite-documentation.html#"
LINKS = {
    "mechanical_model": FDOCS + "StructuralSVK-mechanical_model",
    "CubicCrystal": FDOCS + "StructuralSVK-CubicCrystal",
    "RayleighDamping": FDOCS + "StructuralSVK-RayleighDamping",
    "spectrum": FDOCS + "StructuralSVK-spectrum",
    "probe_dof": FDOCS + "StructuralSVK-probe_dof",
    "build_model": FDOCS + "StructuralSVK-build_model",
    "parametrise": DOCS + "ParametrisationMethod-parametrise",
    "ResonanceConfig": DOCS + "Resonance-ResonanceConfig",
    "observable_polynomial": DOCS + "RomIO-observable_polynomial",
    "cycle_amplitude": DOCS + "RomIO-cycle_amplitude",
    "normal_form_branch": DOCS + "RomIO-normal_form_branch",
    "restrict_ReducedDynamics_to_degree": DOCS
    + "ParametrisationObjects-restrict_ReducedDynamics_to_degree",
    "restrict_polynomial_to_degree": DOCS + "Polynomials-restrict_polynomial_to_degree",
    "save_rom": DOCS + "RomIO-save_rom",
}

KEYWORDS = {"using", "import", "const", "function", "end", "for", "in", "do", "return",
            "if", "elseif", "else", "while", "let", "begin", "struct", "module",
            "true", "false", "nothing"}

# Julia is tokenised only as far as the highlighting needs: comments and strings first,
# then symbols, numbers, identifiers and the operators the page's CSS colours.
TOKEN = re.compile(r"""
    (?P<cm>\#[^\n]*)
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<sym>(?<![A-Za-z0-9_.:]):[A-Za-z_][A-Za-z0-9_!]*)
  | (?P<num>\d+\.?\d*(?:[eE][-+]?\d+)?(?:im)?|\.\d+(?:[eE][-+]?\d+)?)
  | (?P<word>[A-Za-z_À-￿][A-Za-z0-9_!À-￿]*)
  | (?P<op>[=+\-*/^%<>&|]+)
""", re.VERBOSE)


def highlight(code: str) -> str:
    out = []
    pos = 0
    for m in TOKEN.finditer(code):
        out.append(html.escape(code[pos:m.start()]))
        pos = m.end()
        kind = m.lastgroup
        text = html.escape(m.group())
        if kind == "word":
            if m.group() in KEYWORDS:
                kind = "kw"
            elif code[m.end():m.end() + 1] == "(":
                kind = "fn"
            else:
                kind = "var"
        out.append(f'<span class="{kind}">{text}</span>')
    out.append(html.escape(code[pos:]))
    marked = "".join(out)
    for name, href in LINKS.items():
        marked = marked.replace(
            f'<span class="fn">{name}</span>',
            f'<a class="doc-link" href="{href}"><span class="fn">{name}</span></a>')
    return marked


def blocks(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    return ["".join(c["source"]).rstrip("\n")
            for c in nb["cells"] if c["cell_type"] == "code"]


BLOCK = re.compile(r'(<!-- nbcode:(\d+) -->\n)<pre class="code">.*?</pre>', re.S)


def main() -> int:
    check = "--check" in sys.argv[1:]
    page = PAGE.read_text()
    cells = blocks(NB)

    seen = sorted(int(n) for n in re.findall(r"<!-- nbcode:(\d+) -->", page))
    expected = list(range(1, len(cells) + 1))
    if seen != expected:
        print(f"marker mismatch: page has {seen}, notebook has {len(cells)} code cells",
              file=sys.stderr)
        return 2

    def repl(m):
        marker, n = m.group(1), int(m.group(2))
        return f'{marker}<pre class="code">{highlight(cells[n - 1])}</pre>'

    new = BLOCK.sub(repl, page)
    if new == page:
        print(f"{PAGE.name} is up to date ({len(cells)} code blocks)")
        return 0
    if check:
        print(f"{PAGE.name} is OUT OF DATE with {NB.name}", file=sys.stderr)
        return 1
    PAGE.write_text(new)
    print(f"rewrote {len(cells)} code blocks in {PAGE.name} from {NB.name}")
    return 0


sys.exit(main())
