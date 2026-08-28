# Website generation

Run the core API generator from the MORFE.jl repository root:

```sh
julia --project=. website/generate_documentation.jl
```

Generate the MORFEFerrite companion API page from the sibling checkout used during
development:

```sh
julia --project=../MORFEFerrite/MORFEFerrite.jl \
    website/generate_morfeferrite_documentation.jl
```

The companion environment must resolve this MORFE checkout. If necessary, prepare it
once with:

```sh
julia --project=../MORFEFerrite/MORFEFerrite.jl -e \
    'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

Both generated HTML files are tracked for local browsing. The GitHub Pages workflow
rebuilds them from MORFE and the current MORFEFerrite `main` branch before deployment.

## Editing the static documentation manifold

The MORFE code-documentation background is generated from an editable surface
equation. Edit `surface(u, v)` in `generate_manifold_accent.jl`; edit `project(x, y,
z)` there if you also want to change its viewing angle, scale or position. Then run:

```sh
julia website/generate_manifold_accent.jl
```

This rewrites the tracked `assets/manifold-accent.svg`. The generator uses only Julia
standard-library functionality and does not draw a trajectory or a perimeter stroke.
