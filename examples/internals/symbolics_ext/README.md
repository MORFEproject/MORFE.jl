# multiindex_sets — Constructing MultiindexSets

This demo shows how to use the Symbolics.jl package to define `NthOrderModels` and `ExternalSystems` in a symbolic way.

| Section | Shows |
|---------|-------|
| 1 | Full model: two-mass oscilator |
| 3 | Examples for ExternalSystem |

No FEM backend and no solve: the script runs in a couple of seconds against the
repository's root environment.

## How to run

From the repository root:

```bash
julia --project -e 'include("examples/internals/symbolics_ext/main.jl")'
```