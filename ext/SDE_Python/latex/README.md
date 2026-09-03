# Itô vs Stratonovich figures, in LaTeX

Python owns the numbers, LaTeX owns the picture. `Ito_vs_Strat.py --latex` writes plain
tables into `data/`; the figures here are pgfplots source you compile yourself, so every
label, tick and legend entry is typeset by LaTeX in whatever font the document is using.

```
ito_common.tex      shared preamble: packages, colours, line styles, panel sizing
ito_path.tex        (a) sample path + integration grid, (b) the two evaluation points
ito_integrals.tex   (c) the two integrals vs their closed forms, (d) the difference
preview.tex         both figures in a 12.2 cm document, to check them against body text
data/*.dat          generated, do not edit
```

## Regenerating the data

```bash
python3 ../Ito_vs_Strat.py --latex      # writes data/*.dat
```

The dense curves are thinned by `decimate = 10` before export, so LaTeX draws 4001 points
instead of 40001. The integrals themselves are always computed on the full
`dt = 1e-4` path; only the plotting resolution changes.

## Compiling

```bash
latexmk -pdf ito_path.tex ito_integrals.tex     # one cropped PDF each
latexmk -pdf preview.tex                        # both, set in a document
```

Two passes are needed, which `latexmk` handles: the legends are collected with
`legend to name` and recalled with `\ref` below the panels.

## In Overleaf

Upload this whole directory (the `.tex` files and `data/`). Set `ito_path.tex` as the main
document and compile: out comes `ito_path.pdf`, cropped to the figure. Same for
`ito_integrals.tex`. Nothing here loads a font package, so the standalone PDF uses the
class default, Computer Modern.

## In a paper

`preview.tex` is the template. In the preamble:

```latex
\usepackage{standalone}   % default mode=input: the figure body is inputted, so
\input{ito_common}        % it is typeset in this document's font, at this size
```

then, where the figure belongs:

```latex
\begin{figure}[t]
  \centering
  \includestandalone{ito_path}
  \caption{...}
\end{figure}
```

`\includestandalone` inputs the figure body rather than including a PDF, which is what
makes the figure adopt the surrounding document's typeface and size. Do not wrap it in
`\includegraphics[width=...]`: that would scale the type along with the axes. The figure
already sizes itself to `\textwidth`.

If the figures live in a subdirectory, point `\datapath` at it before including them:

```latex
\renewcommand{\datapath}{figures/ito/data/}
```

`ito_common.tex` loads `amsmath`, `pgfplots` (with `groupplots`) and TikZ's `calc`. A
document that already loads them is fine; `\input{ito_common}` after them costs nothing.

## Knobs

All of these are `\providecommand`, so redefining them before `\input{ito_common}` wins.
`\panelsep` and `\panelchrome` are also settable per figure, in the figure body followed
by `\itopanelsizes`, which is how `ito_path.tex` gives itself a tighter gap.

| macro | default | meaning |
|---|---|---|
| `\figwidth` | `\textwidth` | total width the figure fills |
| `\panelsep` | `1.45cm` | gap between the two axis boxes, holds the inner y labels |
| `\panelchrome` | `1.75cm` | width the outer y label, its ticks and the last x tick's overhang take |
| `\panelratio` | `0.78` | height / width of one axis box |
| `\datapath` | `data/` | where the tables live |
| `\Ito`, `\Strat` | `\mathcal{I}`, `\mathcal{S}` | symbols for the two integrals |

`\panelchrome` is measured, not derived: pgfplots cannot know how wide a rotated y label
will be until it has typeset it. If you change the y labels or the font size, compile once,
measure the PDF, and adjust. Being a centimetre out costs nothing but a slightly narrow or
wide figure.

## Compile time

About 15 s per figure, for roughly 13k table rows. If that drags, raise `decimate` in
`Ito_vs_Strat.py` and re-export, or switch on `\usepgfplotslibrary{external}` to cache the
compiled figures.
