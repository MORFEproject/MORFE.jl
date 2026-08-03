# multiindex_sets — Constructing MultiindexSets

A `MultiindexSet` decides **which monomials the DPIM solve computes**. This demo builds
one six ways, from the automatic generators to a spectral truncation, and writes an
interactive lattice view of every set it constructs.

| Section | Shows |
|---------|-------|
| 1 | `all_multiindices_up_to`, `multiindices_with_total_degree`, `all_multiindices_in_box`; GrLex order, `find_in_set`, `indices_in_box_with_bounded_degree` |
| 2 | The same sets built by hand from a vector of exponents — plus the `Matrix{Int}` and `Vector{Vector{Int}}` spellings |
| 3 | Combining conditions: total degree in the master coordinates × per-parameter box in θ (the parametric-ROM shape) |
| 4 | `delete_multiindices` and `filter` — both non-mutating |
| 5 | Bounding the superharmonics `s(α) = ⟨λ, α⟩` by a spectral radius on a damped spectrum, and why that breaks downward closure |
| 6 | Checking a custom set against the `parametrise(...; mset = ...)` contract with `validate_multiindex_set` — all five clauses, including both closures |

No FEM backend and no solve: the script runs in a couple of seconds against the
repository's root environment.

## How to run

From the repository root:

```bash
julia --project -e 'include("examples/internals/multiindex_sets/main.jl")'
```

## Output

Four standalone HTML lattice viewers in `results/figures/`:

| File | Content |
|------|---------|
| `fig1_generators.html` | the three generators plus the set `parametrise` builds by default |
| `fig2_anisotropic.html` | box vs. anisotropic z-total × θ-box, in 3 and 4 variables |
| `fig3_deletion.html` | explicit and predicate deletion, and `filter` as the complement |
| `fig4_spectral.html` | split view: the lattice coloured by `\|s(α)\|` beside the superharmonics in the complex plane — hover a monomial to draw the head-to-tail sum reaching it |

Each file is self-contained — no CDN, no external stylesheet or script — so it opens
straight from the file system. Sets in two variables are drawn as an SVG integer grid
with hover tooltips; sets in three are a point lattice you can drag to orbit; sets in
more expose selectors choosing which coordinates to display.

The viewers embedded in the website live in `website/tutorials/assets/multiindex/`.
Refresh them after changing the script with:

```bash
MORFE_LATTICE_OUT=website/tutorials/assets/multiindex \
  julia --project -e 'include("examples/internals/multiindex_sets/main.jl")'
```
