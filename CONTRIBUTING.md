# Contributing to MORFE.jl

MORFE.jl welcomes contributions to the solver, tests, examples, tutorials, documentation, and engineering applications.

## Start with an issue

Before beginning substantial work, search the [issue tracker](https://github.com/MORFEproject/MORFE.jl/issues). Open an issue to report a bug, propose a feature, discuss an example or benchmark, or describe a research/application collaboration. This lets the team agree on scope and avoids duplicated effort.

## Set up a development checkout

Install Julia 1.10 or later, then clone and instantiate the project:

```bash
git clone https://github.com/MORFEproject/MORFE.jl.git
cd MORFE.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Run the formatter with the pinned CI environment before opening a pull request:

```bash
julia --project=.github/format -e 'using Pkg; Pkg.instantiate()'
julia --project=.github/format -e 'using JuliaFormatter; format(["src", "ext", "test", "examples"]; verbose = true)'
```

Commit any formatting changes produced by the second command.

## Pull requests

- Keep each pull request focused on one issue or cohesive improvement.
- Add or update tests for changed behavior, and run the test suite locally when practical.
- Update user-facing documentation, tutorials, or examples when the public behavior changes.
- Explain the problem, approach, and validation in the pull-request description.

## Research and application collaborations

We welcome collaborations involving nonlinear dynamics, structural mechanics, finite-element backends, and new MORFE applications. Please open an issue describing the research question, model or data involved, expected outcome, and any relevant constraints. The project team will use that discussion to identify the right next step.
