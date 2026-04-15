# MORFE.jl Folder Structure and Module Dependencies

## Folder Structure

```text
src/
├── MORFE.jl                                    # Main package entry point
├── Multiindices.jl                             # Multiindex set utilities
├── Polynomials.jl                              # Dense polynomial representation
├── Realification.jl                            # Complex-to-real polynomial transformation
├── FullOrderModel/
│   ├── MultilinearMaps.jl                      # Multilinear map primitives
│   ├── ExternalSystems.jl                      # External forcing system representation
│   └── FullOrderModel.jl                       # N-th order and first-order ODE models
├── SpectralDecomposition/
│   ├── Eigensolvers.jl                         # Generalised eigenvalue solver (ARPACK)
│   ├── EigenModesPropagation.jl                # Left/right eigenvector and Jordan vector propagation
│   └── JordanChain.jl                          # Jordan chain computation (standalone utility)
└── ParametrisationMethod/
    ├── Resonance.jl                            # Resonance detection and resonance sets
    ├── InvarianceEquation.jl                   # Cohomological equation assembly
    ├── MasterModeOrthogonality.jl              # Orthogonality condition assembly
    ├── ParametrisationMethod.jl                # Core types (Parametrisation, ReducedDynamics)
    └── RightHandSide/
        ├── MultilinearTerms.jl                 # Nonlinear multilinear term computation
        └── LowerOrderCouplings.jl              # Lower-order coupling terms
```

## Dependency Graph

```text
MORFE.jl (root)
├── Multiindices.jl          [standalone — no internal dependencies]
├── Polynomials.jl           ← Multiindices.jl
├── FullOrderModel/
│   ├── MultilinearMaps.jl   [standalone — no internal dependencies]
│   ├── ExternalSystems.jl   ← Multiindices.jl, Polynomials.jl
│   └── FullOrderModel.jl    ← Polynomials.jl, MultilinearMaps.jl, ExternalSystems.jl
├── SpectralDecomposition/
│   ├── Eigensolvers.jl      [standalone — no internal dependencies]
│   └── EigenModesPropagation.jl  ← FullOrderModel.jl, ParametrisationMethod.jl
│   (JordanChain.jl — not included in MORFE.jl; standalone utility)
├── Realification.jl         ← Polynomials.jl
└── ParametrisationMethod/
    ├── Resonance.jl         ← Multiindices.jl
    ├── InvarianceEquation.jl  [no internal dependencies; uses LinearAlgebra, StaticArrays]
    ├── MasterModeOrthogonality.jl  [no internal dependencies; uses LinearAlgebra, StaticArrays]
    ├── ParametrisationMethod.jl  ← Multiindices.jl, Polynomials.jl
    └── RightHandSide/
        ├── MultilinearTerms.jl  ← Multiindices.jl, ParametrisationMethod.jl, FullOrderModel.jl
        └── LowerOrderCouplings.jl  ← Multiindices.jl, Polynomials.jl, ParametrisationMethod.jl
```

## Module Details

### 1. Multiindices.jl

- **Purpose**: Multiindex set data structure and utilities for graded lexicographic ordering
- **External dependencies**: `StaticArrays`
- **Exports**: `MultiindexSet`, `zero_multiindex`, `all_multiindices_up_to`, `multiindices_with_total_degree`, `all_multiindices_in_box`, `indices_in_box_with_bounded_degree`, `divides`, `is_constant`, `find_in_set`, `build_exponent_index_map`, `factorisations_asymmetric`, `factorisations_fully_symmetric`, `factorisations_groupwise_symmetric`

### 2. Polynomials.jl

- **Purpose**: Dense polynomial representation with aligned multiindex sets
- **External dependencies**: `LinearAlgebra`, `StaticArrays`
- **Internal imports**: `Multiindices`
- **Exports**: `DensePolynomial`, `polynomial_from_dict`, `polynomial_from_pairs`, `coeffs`, `multiindex_set`, `nvars`, `all_multiindices_up_to`, `coefficient`, `has_term`, `find_term`, `find_in_multiindex_set`, `zero`, `evaluate`, `extract_component`, `each_term`, `similar_poly`

### 3. FullOrderModel/MultilinearMaps.jl

- **Purpose**: Multilinear map primitives used to encode nonlinear terms in the full-order model
- **External dependencies**: none
- **Exports**: `MultilinearMap`, `evaluate_term!`

### 4. FullOrderModel/ExternalSystems.jl

- **Purpose**: Represents a polynomial external forcing system with its linear eigenstructure
- **External dependencies**: `StaticArrays`, `LinearAlgebra`
- **Internal imports**: `Multiindices`, `Polynomials`
- **Exports**: `ExternalSystem`

### 5. FullOrderModel/FullOrderModel.jl

- **Purpose**: N-th order and first-order dynamical system representations
- **External dependencies**: `LinearAlgebra`, `SparseArrays`, `StaticArrays`
- **Internal imports**: `Polynomials`, `MultilinearMaps`, `ExternalSystems`
- **Exports**: `NDOrderModel`, `FirstOrderModel`, `linear_first_order_matrices`, `evaluate_nonlinear_terms!`

### 6. SpectralDecomposition/Eigensolvers.jl

- **Purpose**: Generalised eigenvalue problem solver using ARPACK with optional shift-and-invert
- **External dependencies**: `Arpack`, `LinearAlgebra`, `LinearMaps`, `SparseArrays`
- **Exports**: `generalised_eigenpairs`

### 7. SpectralDecomposition/EigenModesPropagation.jl

- **Purpose**: Propagate left/right eigenvectors and Jordan vectors through the companion first-order system arising from an N-th order ODE
- **Internal imports**: `FullOrderModel`, `ParametrisationMethod`
- **Exports**: `propagate_left_eigenvector_from_last`, `propagate_left_jordan_vector`, `propagate_right_eigenvector_form_first`, `propagate_right_jordan_vector`

### 8. SpectralDecomposition/JordanChain.jl

- **Purpose**: Compute Jordan chains for the generalised eigenvalue problem `A v = λ B v`; standalone utility not included in `MORFE.jl`
- **External dependencies**: `LinearAlgebra`, `SparseArrays`, `Printf`
- **Exports**: `compute_jordan_chain`

### 9. Realification.jl

- **Purpose**: Transform complex variable polynomials (z, z̄) to real variable polynomials (x, y) where z = x + iy; coefficients remain complex-valued
- **External dependencies**: `LinearAlgebra`, `StaticArrays`
- **Internal imports**: `Polynomials`
- **Exports**: `realify`, `compose_linear`, `realify_via_linear`

### 10. ParametrisationMethod/Resonance.jl

- **Purpose**: Resonance detection and `ResonanceSet` construction supporting graph-style, complex/real normal form, and condition-number–based strategies
- **Internal imports**: `Multiindices`
- **Exports**: `ResonanceSet`, `resonance_set_from_graph_style`, `resonance_set_from_complex_normal_form_style`, `resonance_set_from_real_normal_form_style`, `resonance_set_from_condition_number_estimate`, `empty_resonance_set`, `set_resonance!`, `is_resonant`, `resonant_targets`, `resonant_multiindices`, `EigenvalueCondition`, `RealEigenvalueCondition`, `ConditionNumberEstimateCondition`, `GraphInternal`, `NormalFormInternal`, `ResonanceStyle`

### 11. ParametrisationMethod/InvarianceEquation.jl

- **Purpose**: Assemble the cohomological linear systems of the parametrisation method using fused Horner passes; supports external forcing
- **External dependencies**: `LinearAlgebra`, `StaticArrays`
- **Exports**: `precompute_column_polynomials`, `evaluate_system_matrix_and_lower_order_rhs!`, `evaluate_column!`, `evaluate_external_rhs!`, `assemble_cohomological_matrix_and_rhs`

### 12. ParametrisationMethod/MasterModeOrthogonality.jl

- **Purpose**: Assemble the orthogonality conditions (w.r.t. resonant master modes) that complement the cohomological equation
- **External dependencies**: `LinearAlgebra`, `StaticArrays`
- **Exports**: `precompute_orthogonality_operator_coefficients`, `precompute_orthogonality_column_polynomials`, `evaluate_orthogonality_row_and_lower_order_rhs!`, `evaluate_orthogonality_column_row!`, `evaluate_orthogonality_external_rhs`, `assemble_orthogonality_matrix_and_rhs`

### 13. ParametrisationMethod/ParametrisationMethod.jl

- **Purpose**: Core types for the parametrisation method: `Parametrisation` (mapping from reduced coordinates to full state) and `ReducedDynamics`
- **External dependencies**: `StaticArrays`
- **Internal imports**: `Multiindices`, `Polynomials`
- **Exports**: `Parametrisation`, `ReducedDynamics`, `create_parametrisation_method_objects`

### 14. ParametrisationMethod/RightHandSide/MultilinearTerms.jl

- **Purpose**: Compute nonlinear multilinear term contributions for the right-hand side of the ROM equations, with caching support
- **Internal imports**: `Multiindices`, `ParametrisationMethod`, `FullOrderModel`
- **Exports**: `compute_multilinear_terms`, `build_multilinear_terms_cache`, `MultilinearTermsCache`

### 15. ParametrisationMethod/RightHandSide/LowerOrderCouplings.jl

- **Purpose**: Compute lower-order coupling terms in the parametrisation method
- **External dependencies**: `LinearAlgebra`, `StaticArrays`
- **Internal imports**: `Multiindices`, `Polynomials`, `ParametrisationMethod`
- **Exports**: `compute_lower_order_couplings`

## Public API (re-exported from MORFE.jl)

```julia
using .Multiindices
using .Polynomials: DensePolynomial, evaluate
using .MultilinearMaps
using .ExternalSystems
using .FullOrderModel
using .Eigensolvers: generalised_eigenpairs
using .Resonance
using .InvarianceEquation
using .MasterModeOrthogonality
using .ParametrisationMethod
using .EigenModesPropagation
using .MultilinearTerms: compute_multilinear_terms
using .LowerOrderCouplings

export MultiindexSet, zero_multiindex,
       all_multiindices_up_to, multiindices_with_total_degree,
       all_multiindices_in_box, indices_in_box_with_bounded_degree
export DensePolynomial, evaluate
export MultilinearMap, ExternalSystem
export FullOrderModel, FirstOrderModel, NDOrderModel,
       linear_first_order_matrices, evaluate_nonlinear_terms!
export SingleResonance, ResonanceSet, resonance_set, resonance_set_from_eigenvalues
export Parametrisation, create_parametrisation_method_objects
export compute_multilinear_terms
```

## Dependency Summary Table

| Module | External Dependencies | Internal Dependencies |
| --- | --- | --- |
| Multiindices.jl | StaticArrays | — |
| Polynomials.jl | LinearAlgebra, StaticArrays | Multiindices.jl |
| MultilinearMaps.jl | — | — |
| ExternalSystems.jl | StaticArrays, LinearAlgebra | Multiindices.jl, Polynomials.jl |
| FullOrderModel.jl | LinearAlgebra, SparseArrays, StaticArrays | Polynomials.jl, MultilinearMaps.jl, ExternalSystems.jl |
| Eigensolvers.jl | Arpack, LinearAlgebra, LinearMaps, SparseArrays | — |
| EigenModesPropagation.jl | — | FullOrderModel.jl, ParametrisationMethod.jl |
| JordanChain.jl | LinearAlgebra, SparseArrays, Printf | — |
| Realification.jl | LinearAlgebra, StaticArrays | Polynomials.jl |
| Resonance.jl | — | Multiindices.jl |
| InvarianceEquation.jl | LinearAlgebra, StaticArrays | — |
| MasterModeOrthogonality.jl | LinearAlgebra, StaticArrays | — |
| ParametrisationMethod.jl | StaticArrays | Multiindices.jl, Polynomials.jl |
| MultilinearTerms.jl | — | Multiindices.jl, ParametrisationMethod.jl, FullOrderModel.jl |
| LowerOrderCouplings.jl | LinearAlgebra, StaticArrays | Multiindices.jl, Polynomials.jl, ParametrisationMethod.jl |
