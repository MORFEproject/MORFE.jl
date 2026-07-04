# Graph Report - .  (2026-07-02)

## Corpus Check
- Large corpus: 539 files · ~3,077,608 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 444 nodes · 454 edges · 63 communities (55 shown, 8 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Eigensolvers|Eigensolvers]]
- [[_COMMUNITY_Cohomological Equations|Cohomological Equations]]
- [[_COMMUNITY_MORFE Module Registry|MORFE Module Registry]]
- [[_COMMUNITY_SDE Integrator|SDE Integrator]]
- [[_COMMUNITY_Multiindex Arithmetic|Multiindex Arithmetic]]
- [[_COMMUNITY_Resonance Detection|Resonance Detection]]
- [[_COMMUNITY_Full Order Model|Full Order Model]]
- [[_COMMUNITY_SVK High-Level API|SVK High-Level API]]
- [[_COMMUNITY_Dense Polynomials|Dense Polynomials]]
- [[_COMMUNITY_Abaqus-Gmsh Mesh IO|Abaqus-Gmsh Mesh IO]]
- [[_COMMUNITY_Invariance Error Validation|Invariance Error Validation]]
- [[_COMMUNITY_Gmsh Extensions|Gmsh Extensions]]
- [[_COMMUNITY_Arpack Extension|Arpack Extension]]
- [[_COMMUNITY_VTK Export Extension|VTK Export Extension]]
- [[_COMMUNITY_BifurcationKit Extension|BifurcationKit Extension]]
- [[_COMMUNITY_Plots Extension|Plots Extension]]
- [[_COMMUNITY_Symbolic Extraction|Symbolic Extraction]]
- [[_COMMUNITY_Benchmark Logging|Benchmark Logging]]
- [[_COMMUNITY_Parametrisation Types|Parametrisation Types]]
- [[_COMMUNITY_Lower Order Couplings|Lower Order Couplings]]
- [[_COMMUNITY_Multilinear RHS Terms|Multilinear RHS Terms]]
- [[_COMMUNITY_Realification|Realification]]
- [[_COMMUNITY_Gmsh-to-COMSOL Export|Gmsh-to-COMSOL Export]]
- [[_COMMUNITY_Ferrite Extension|Ferrite Extension]]
- [[_COMMUNITY_Jordan Chain Solver|Jordan Chain Solver]]
- [[_COMMUNITY_Convenience API|Convenience API]]
- [[_COMMUNITY_Tensor Symmetry Types|Tensor Symmetry Types]]
- [[_COMMUNITY_COMSOL-to-Gmsh Import|COMSOL-to-Gmsh Import]]
- [[_COMMUNITY_ParaView Export|ParaView Export]]
- [[_COMMUNITY_External Forcing Systems|External Forcing Systems]]
- [[_COMMUNITY_Eigenpair Computation|Eigenpair Computation]]
- [[_COMMUNITY_Pardiso Sparse Solver|Pardiso Sparse Solver]]
- [[_COMMUNITY_Symbolics Extension|Symbolics Extension]]
- [[_COMMUNITY_Eigenmode Propagation|Eigenmode Propagation]]
- [[_COMMUNITY_Invariance Equation|Invariance Equation]]
- [[_COMMUNITY_FEM Cached RHS Replay|FEM Cached RHS Replay]]
- [[_COMMUNITY_COMSOL-Ferrite Mesh Loader|COMSOL-Ferrite Mesh Loader]]
- [[_COMMUNITY_Symbolic External System|Symbolic External System]]
- [[_COMMUNITY_SVK Postprocessing|SVK Postprocessing]]
- [[_COMMUNITY_SVK Material Types|SVK Material Types]]
- [[_COMMUNITY_Conjugate Symmetry|Conjugate Symmetry]]
- [[_COMMUNITY_Master Mode Orthogonality|Master Mode Orthogonality]]
- [[_COMMUNITY_Cached Split Structs|Cached Split Structs]]
- [[_COMMUNITY_BifurcationKit Interface|BifurcationKit Interface]]
- [[_COMMUNITY_Multilinear Maps|Multilinear Maps]]
- [[_COMMUNITY_SVK Mechanical Model|SVK Mechanical Model]]
- [[_COMMUNITY_FEM Utility|FEM Utility]]

## God Nodes (most connected - your core abstractions)
1. `CohomologicalEquations` - 22 edges
2. `MORFE` - 21 edges
3. `Eigenproblems` - 19 edges
4. `Multiindices` - 18 edges
5. `SDEIntegrator` - 15 edges
6. `Resonance` - 15 edges
7. `MORFEStructuralSVK` - 13 edges
8. `FullOrderModel` - 12 edges
9. `Polynomials` - 11 edges
10. `InvarianceError` - 10 edges

## Surprising Connections (you probably didn't know these)
- `RayleighEigenSolver` --inherits--> `AbstractEigensolver`  [EXTRACTED]
  ext/StructuralSVK/rayleigh_solver.jl → src/SpectralDecomposition/Eigenproblems.jl

## Import Cycles
- None detected.

## Communities (63 total, 8 thin omitted)

### Community 0 - "Eigensolvers"
Cohesion: 0.10
Nodes (18): Float64, Int, RayleighEigenSolver, Int64, AbstractEigensolver, ArpackEigensolver, DefaultEigensolver, Eigenproblems (+10 more)

### Community 1 - "Cohomological Equations"
Cohesion: 0.09
Nodes (20): KLU, CohomologicalEquations, Bool, FullOrderModel, Int, InvarianceEquation, LinearAlgebra, LowerOrderCouplings (+12 more)

### Community 2 - "MORFE Module Registry"
Cohesion: 0.09
Nodes (21): BifurcationKitInterface, CohomologicalEquations, ConvenienceMethods, FEMUtility, InvarianceError, ParaviewExport, Realification, Eigenproblems (+13 more)

### Community 3 - "SDE Integrator"
Cohesion: 0.15
Nodes (3): Parameters         ----------         x0 : array_like, shape (d,)         t_span, Integrator for Stratonovich or Itô SDEs with diagonal/scalar noise.      By defa, SDEIntegrator

### Community 4 - "Multiindex Arithmetic"
Cohesion: 0.15
Nodes (13): all_multiindices_in_box(), all_multiindices_up_to(), FactorisationEntry, _generate_ascending_lex_fixed!(), _grlex_rank(), Int, StaticArrays, _last_index_below_degree() (+5 more)

### Community 5 - "Resonance Detection"
Cohesion: 0.18
Nodes (15): apply_internal_resonances!(), _build_inner_matrix(), _build_outer_matrix(), ConditionNumberEstimateCondition, EigenvalueCondition, GraphInternal, InternalResonance, Float64 (+7 more)

### Community 6 - "Full Order Model"
Cohesion: 0.14
Nodes (12): MT, AbstractFullOrderModel, FullOrderModel, _info_implicit_symmetry(), ExternalSystems, Int, LinearAlgebra, MultilinearMaps (+4 more)

### Community 7 - "SVK High-Level API"
Cohesion: 0.14
Nodes (11): Arpack, Ferrite, LinearAlgebra, LinearMaps, MORFE, Printf, SparseArrays, StaticArrays (+3 more)

### Community 8 - "Dense Polynomials"
Cohesion: 0.17
Nodes (7): Base, Base.Threads, Mmap, LinearAlgebra, Multiindices, StaticArrays, Polynomials

### Community 9 - "Abaqus-Gmsh Mesh IO"
Cohesion: 0.27
Nodes (11): abaqus_to_gmsh(), abaqus_to_gmsh_linear(), AbaqusToGmsh, _build_and_write(), _ElemSection, Gmsh, Int, _parse_keyword() (+3 more)

### Community 10 - "Invariance Error Validation"
Cohesion: 0.18
Nodes (7): Random, InvarianceError, FullOrderModel, LinearAlgebra, ParametrisationMethod, Polynomials, Statistics

### Community 11 - "Gmsh Extensions"
Cohesion: 0.20
Nodes (8): AbaqusToGmsh, ComsolToGmsh, Gmsh, MORFE, Printf, MORFEGmshExt, GmshToComsol, MORFE.FEMUtility

### Community 12 - "Arpack Extension"
Cohesion: 0.20
Nodes (8): Arpack, LinearAlgebra, LinearMaps, MORFE, SparseArrays, MORFEArpackExt, MORFE.Eigenproblems, MORFE.Eigensolvers

### Community 13 - "VTK Export Extension"
Cohesion: 0.22
Nodes (7): Ferrite, MORFE, MORFE.ParametrisationMethod, Printf, MORFEWriteVTKExt, MORFE.ParaviewExport, WriteVTK

### Community 14 - "BifurcationKit Extension"
Cohesion: 0.25
Nodes (7): BifurcationKit, LinearAlgebra, MORFE, MORFE.ParametrisationMethod, MORFEBifurcationKitExt, MORFE.BifurcationKitInterface, MORFE.Polynomials

### Community 15 - "Plots Extension"
Cohesion: 0.29
Nodes (7): MORFE, Statistics, MORFEPlotsExt, _plot_convergence(), _reference_line_params(), MORFE.InvarianceError, Plots

### Community 16 - "Symbolic Extraction"
Cohesion: 0.39
Nodes (4): degree_of_monomial(), _get_taylor_expansion_around_0(), seperate_into_monomials(), _to_MyNum()

### Community 17 - "Benchmark Logging"
Cohesion: 0.29
Nodes (3): Float64, Int, _OrderAccum

### Community 18 - "Parametrisation Types"
Cohesion: 0.25
Nodes (5): LinearAlgebra, Multiindices, Polynomials, StaticArrays, ParametrisationMethod

### Community 19 - "Lower Order Couplings"
Cohesion: 0.25
Nodes (6): LinearAlgebra, Multiindices, ParametrisationMethod, Polynomials, StaticArrays, LowerOrderCouplings

### Community 20 - "Multilinear RHS Terms"
Cohesion: 0.25
Nodes (7): FullOrderModel, LinearAlgebra, Multiindices, MultilinearMaps, ParametrisationMethod, StaticArrays, MultilinearTerms

### Community 21 - "Realification"
Cohesion: 0.25
Nodes (4): LinearAlgebra, Polynomials, StaticArrays, Realification

### Community 22 - "Gmsh-to-COMSOL Export"
Cohesion: 0.38
Nodes (6): gmsh_to_comsol(), GmshToComsol, Gmsh, Printf, _write_elem_section(), _write_header()

### Community 24 - "Ferrite Extension"
Cohesion: 0.29
Nodes (5): Ferrite, LinearAlgebra, MORFE, SparseArrays, MORFEFerriteExt

### Community 25 - "Jordan Chain Solver"
Cohesion: 0.29
Nodes (4): LinearAlgebra, Printf, SparseArrays, JordanChain

### Community 26 - "Convenience API"
Cohesion: 0.29
Nodes (5): ConvenienceMethods, Eigenproblems, FullOrderModel, Multiindices, Resonance

### Community 27 - "Tensor Symmetry Types"
Cohesion: 0.57
Nodes (5): FullyAsymmetric, FullySymmetric, GroupwiseSymmetric, symmetry_type(), SymmetryType

### Community 28 - "COMSOL-to-Gmsh Import"
Cohesion: 0.47
Nodes (5): comsol_to_gmsh(), comsol_to_gmsh_linear(), ComsolToGmsh, Gmsh, _read_mesh()

### Community 30 - "External Forcing Systems"
Cohesion: 0.33
Nodes (5): ExternalSystems, LinearAlgebra, Multiindices, Polynomials, StaticArrays

### Community 31 - "Eigenpair Computation"
Cohesion: 0.33
Nodes (3): Eigensolvers, LinearAlgebra, SparseArrays

### Community 32 - "Pardiso Sparse Solver"
Cohesion: 0.40
Nodes (4): MORFE, MORFEPardisoExt, MORFE.CohomologicalEquations, Pardiso

### Community 33 - "Symbolics Extension"
Cohesion: 0.40
Nodes (4): MORFE, StaticArrays, MORFESymbolicsExt, Symbolics

### Community 34 - "Eigenmode Propagation"
Cohesion: 0.40
Nodes (4): LinearAlgebra, MORFE.ParametrisationMethod, PropagateEigenmodes, MORFE.FullOrderModel

### Community 35 - "Invariance Equation"
Cohesion: 0.40
Nodes (4): InvarianceEquation, LinearAlgebra, SparseArrays, StaticArrays

### Community 36 - "FEM Cached RHS Replay"
Cohesion: 0.60
Nodes (4): _accumulate_global_entries!(), _replay_all_fem_splits!(), _replay_split!(), _replay_term!()

### Community 37 - "COMSOL-Ferrite Mesh Loader"
Cohesion: 0.83
Nodes (3): load_comsol_grid(), QuadraticWedge, _read_mesh()

### Community 41 - "SVK Material Types"
Cohesion: 1.00
Nodes (3): HarmonicForcing(), RayleighDamping(), SVKMaterial()

### Community 44 - "Master Mode Orthogonality"
Cohesion: 0.50
Nodes (3): LinearAlgebra, StaticArrays, MasterModeOrthogonality

### Community 46 - "Cached Split Structs"
Cohesion: 0.50
Nodes (3): CachedSplit, Bool, Int

## Knowledge Gaps
- **151 isolated node(s):** `Gmsh`, `Gmsh`, `Gmsh`, `Printf`, `MORFE` (+146 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `Gmsh`, `Gmsh`, `Gmsh` to the rest of the system?**
  _153 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Eigensolvers` be split into smaller, more focused modules?**
  _Cohesion score 0.10153846153846154 - nodes in this community are weakly interconnected._
- **Should `Cohomological Equations` be split into smaller, more focused modules?**
  _Cohesion score 0.09 - nodes in this community are weakly interconnected._
- **Should `MORFE Module Registry` be split into smaller, more focused modules?**
  _Cohesion score 0.09090909090909091 - nodes in this community are weakly interconnected._
- **Should `SDE Integrator` be split into smaller, more focused modules?**
  _Cohesion score 0.14736842105263157 - nodes in this community are weakly interconnected._
- **Should `Multiindex Arithmetic` be split into smaller, more focused modules?**
  _Cohesion score 0.14736842105263157 - nodes in this community are weakly interconnected._
- **Should `Full Order Model` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._