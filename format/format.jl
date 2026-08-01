# Format the MORFE.jl sources with the SciML style defined in `.JuliaFormatter.toml`.
#
#   julia --project=format format/format.jl        # format in place
#
# `format` returns `true` when nothing needed changing, so the same invocation
# doubles as the CI gate: the workflow runs this, then `git diff --exit-code`
# shows the contributor exactly what to apply.
#
# The JuliaFormatter version is pinned exactly in format/Project.toml and its
# Manifest.toml is committed, so local runs and CI produce identical output.

using JuliaFormatter

const ROOT = dirname(@__DIR__)
const TARGETS = ["src", "ext", "test"]

paths = [joinpath(ROOT, dir) for dir in TARGETS]
already_formatted = format(paths; verbose = true)

if already_formatted
    @info "All files already formatted." targets = TARGETS
else
    @warn "Files were reformatted. Review and commit the changes." targets = TARGETS
    exit(1)
end
