# website/generate_morfeferrite_documentation.jl
#
# From a sibling checkout:
#   julia --project=../MORFEFerrite/MORFEFerrite.jl \
#       website/generate_morfeferrite_documentation.jl
#
# In CI, MORFEFerrite.jl is checked out inside this repository and used as the active
# project. `pathof` makes source links work in either layout.

using MORFEFerrite

const MORFEFERRITE_REPO_ROOT = normpath(joinpath(dirname(pathof(MORFEFerrite)), ".."))

const MORFE_DOC_CONFIG = (
    package_name = "MORFEFerrite.jl",
    page_title = "Code Documentation — MORFEFerrite.jl",
    output_name = "morfeferrite-documentation.html",
    repo_root = MORFEFERRITE_REPO_ROOT,
    github_repo = "https://github.com/MORFEproject/MORFEFerrite.jl",
    github_base = "https://github.com/MORFEproject/MORFEFerrite.jl/blob/main",
    companion = true,
    modules = [
        (MORFEFerrite.Common, "Common"),
        (MORFEFerrite.Common.MeshIO, "MeshIO"),
        (MORFEFerrite.ParametricGeometry, "ParametricGeometry"),
        (MORFEFerrite.StructuralSVK, "StructuralSVK"),
        (MORFEFerrite.FluidNavierStokes, "FluidNavierStokes")
    ]
)

include("generate_documentation.jl")
