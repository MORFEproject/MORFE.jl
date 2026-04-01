# MORFE.jl Folder Structure and Module Dependencies

## Folder Structure

```
src/
├── MORFE.jl                              # Main package entry point
├── Multiindices.jl                       # Multiindex set utilities
├── Polynomials.jl                         # Dense polynomial representation
├── Eigensolvers.jl                        # Generalized eigenvalue solver
├── FullOrderModel.jl                      # N-th order and first-order ODE models
├── Realification.jl                       # Complex-to-real polynomial transformation
└── ParametrisationMethod/
    ├── ParametrisationMethod.jl           # Core types (Parametrisation, ReducedDynamics)
    ├── LinearOperator/                    # (empty directory)
    └── RightHandSide/
        ├── MultilinearTerms.jl           # Nonlinear term computation
        └── LowerOrderCouplings.jl         # Lower-order coupling terms
```

## Dependency Graph

```
MORFE.jl (root)
├── Multiindices.jl          [standalone - no internal dependencies]
├── Polynomials.jl           ← Multiindices.jl
├── FullOrderModel.jl        [standalone - no internal dependencies]
├── Eigensolvers.jl          [standalone - no internal dependencies]
├── Realification.jl          ← Polynomials.jl
└── ParametrisationMethod/
    └── ParametrisationMethod.jl
        ├── ← Multiindices.jl
        └── ← Polynomials.jl
            └── ← Multiindices.jl
    └── RightHandSide/
        ├── MultilinearTerms.jl
        │   ├── ← Multiindices.jl
        │   ├── ← ParametrisationMethod.jl
        │   └── ← FullOrderModel.jl
        └── LowerOrderCouplings.jl
            ├── ← Multiindices.jl
            ├── ← Polynomials.jl
            └── ← ParametrisationMethod.jl
```

## Module Details

### 1. Multiindices.jl
- **Purpose**: Multiindex set data structure and utilities for graded lexicographic ordering
- **External dependencies**: `StaticArrays`
- **Exports**: `MultiindexSet`, `zero_multiindex`, `all_multiindices_up_to`, `multiindices_with_total_degree`, `all_multiindices_in_box`, `indices_in_box_with_bounded_degree`, `divides`, `is_constant`, `find_in_set`, `build_exponent_index_map`, `factorisations_asymmetric`, `factorisations_fully_symmetric`, `factorisations_groupwise_symmetric`

### 2. Polynomials.jl
- **Purpose**: Dense polynomial representation with aligned multiindex sets
- **External dependencies**: `LinearAlgebra`, `StaticArrays`
- **Internal imports**: `Multiindices` (`MultiindexSet`, `find_in_set`, `zero_multiindex`)
- **Exports**: `DensePolynomial`, `polynomial_from_dict`, `polynomial_from_pairs`, `coeffs`, `multiindex_set`, `nvars`, `all_multiindices_up_to`, `coefficient`, `has_term`, `find_term`, `find_in_multiindex_set`, `zero`, `evaluate`, `extract_component`, `each_term`, `similar_poly`

### 3. FullOrderModel.jl
- **Purpose**: N-th order and first-order dynamical system representations
- **External dependencies**: `LinearAlgebra`, `SparseArrays`
- **Exports**: `NDOrderModel`, `FirstOrderModel`, `MultilinearMap`, `linear_first_order_matrices`, `evaluate_nonlinear_terms!`

### 4. Eigensolvers.jl
- **Purpose**: Generalized eigenvalue problem solver using ARPACK
- **External dependencies**: `Arpack`, `LinearAlgebra`, `LinearMaps`, `SparseArrays`
- **Exports**: `generalized_eigenpairs`

### 5. Realification.jl
- **Purpose**: Transform complex variable polynomials (z, z̄) to real variable polynomials (x, y) where z = x + iy and z̄ = x - iy; coefficients remain complex-valued
- **External dependencies**: `LinearAlgebra`, `StaticArrays`
- **Internal imports**: `Polynomials` (`DensePolynomial`, `nvars`, `each_term`, `similar_poly`, `coefficient`, `coeffs`, `multiindex_set`)
- **Exports**: `realify`, `compose_linear`, `realify_via_linear`

### 6. ParametrisationMethod.jl
- **Purpose**: Core types for the parametrisation method (Reduced Order Modeling)
- **External dependencies**: `StaticArrays`
- **Internal imports**: `Multiindices` (`MultiindexSet`), `Polynomials` (`DensePolynomial`)
- **Exports**: `Parametrisation` (type alias), `ReducedDynamics` (type alias), `create_parametrisation_method_objects`

### 7. MultilinearTerms.jl
- **Purpose**: Compute nonlinear multilinear terms contributions for the ROM
- **Internal imports**: `Multiindices` (`indices_in_box_with_bounded_degree`, factorisation functions), `ParametrisationMethod` (`Parametrisation`), `FullOrderModel` (`NDOrderModel`, `FirstOrderModel`, `MultilinearMap`)
- **Exports**: `compute_multilinear_terms`

### 8. LowerOrderCouplings.jl
- **Purpose**: Compute lower-order coupling terms in the parametrisation method
- **External dependencies**: `LinearAlgebra`, `StaticArrays`
- **Internal imports**: `Multiindices` (`MultiindexSet`, `indices_in_box_with_bounded_degree`, `build_exponent_index_map`), `Polynomials` (`coeffs`, `multiindex_set`, `nvars`), `ParametrisationMethod` (`Parametrisation`, `ReducedDynamics`)
- **Exports**: `compute_lower_order_couplings`

## Public API (re-exported from MORFE.jl)

```julia
using .Multiindices          # All exports from Multiindices.jl
using .Polynomials: DensePolynomial, evaluate
using .FullOrderModel: FullOrderModel, FirstOrderModel, NDOrderModel
using .Eigensolvers: generalized_eigenpairs
using .ParametrisationMethod  # All exports from ParametrisationMethod.jl
using .MultilinearTerms: compute_multilinear_terms
using .LowerOrderCouplings   # All exports from LowerOrderCouplings.jl

export MultiindexSet
export DensePolynomial, evaluate
export FullOrderModel, FirstOrderModel, NDOrderModel
export Parametrisation
export compute_multilinear_terms
```

## Dependency Summary Table

| Module | External Dependencies | Internal Dependencies |
|--------|---------------------|---------------------|
| Multiindices.jl | StaticArrays | — |
| Polynomials.jl | LinearAlgebra, StaticArrays | Multiindices.jl |
| FullOrderModel.jl | LinearAlgebra, SparseArrays | — |
| Eigensolvers.jl | Arpack, LinearAlgebra, LinearMaps, SparseArrays | — |
| Realification.jl | LinearAlgebra, StaticArrays | Polynomials.jl |
| ParametrisationMethod.jl | StaticArrays | Multiindices.jl, Polynomials.jl |
| MultilinearTerms.jl | — | Multiindices.jl, ParametrisationMethod.jl, FullOrderModel.jl |
| LowerOrderCouplings.jl | LinearAlgebra, StaticArrays | Multiindices.jl, Polynomials.jl, ParametrisationMethod.jl |
