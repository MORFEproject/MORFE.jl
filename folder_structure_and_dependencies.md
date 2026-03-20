# MORFE Source Code Structure and Dependencies

## Folder Structure

```
src/
├── MORFE.jl                          # Main package entry point
├── FullOrderModel.jl                # Core model types
├── Multiindices.jl                  # Multiindex set data structure
├── Polynomials.jl                   # Polynomial representations
├── Realification.jl                 # Complex-to-real polynomial transformation
├── Eigensolvers.jl                  # Generalized eigenvalue solver
├── deprecated_InputFullOrderModel/   # Deprecated input modules
│   ├── InputModel.jl
│   ├── InputModelByHand.jl
│   ├── InputModelGridapMechanical.jl
│   ├── InputModelInterface.jl
│   └── Morfe_2_0/
└── ParametrisationMethod/
    ├── Parametrisation.jl           # Parametrisation type alias
    └── RightHandSide/
        ├── MultilinearTerms.jl      # Multilinear term computation
        └── LowerOrderCouplings.jl  # Lower-order coupling computation
```

## Module Dependencies

### Dependency Graph

```
MORFE.jl
    ├── Multiindices.jl
    ├── Polynomials.jl
    │       └── (uses Multiindices.jl)
    ├── FullOrderModel.jl
    │       └── (uses LinearAlgebra, SparseArrays)
    ├── Eigensolvers.jl
    │       └── (uses Arpack, LinearAlgebra, LinearMaps, SparseArrays)
    ├── Realification.jl
    │       └── (uses Polynomials.jl)
    └── ParametrisationMethod/
            ├── Parametrisation.jl
            │       └── (uses Polynomials.jl)
            └── RightHandSide/
                    ├── MultilinearTerms.jl
                    │       ├── (uses Multiindices.jl)
                    │       ├── (uses Polynomials.jl)
                    │       ├── (uses Parametrisation.jl)
                    │       └── (uses FullOrderModel.jl)
                    └── LowerOrderCouplings.jl
                            ├── (uses Multiindices.jl)
                            └── (uses Polynomials.jl)
```

### Detailed Module Imports

#### 1. MORFE.jl (Main Package)
- **Includes**: `Multiindices.jl`, `Polynomials.jl`, `FullOrderModel.jl`, `Eigensolvers.jl`, `Realification.jl`, `Parametrisation.jl`, `MultilinearTerms.jl`, `LowerOrderCouplings.jl`
- **Exports**:
  - `MultiindexSet` (from `Multiindices`)
  - `DensePolynomial`, `evaluate` (from `Polynomials`)
  - `FullOrderModel`, `FirstOrderModel`, `NDOrderModel` (from `FullOrderModel`)
  - `Parametrisation` (from `Parametrisation`)
  - `compute_multilinear_terms` (from `MultilinearTerms`)

#### 2. Multiindices.jl
- **External dependencies**: None (standalone)
- **Exports**:
  - `MultiindexSet`
  - `zero_multiindex`, `multiindex`
  - `all_multiindices_up_to`, `multiindices_with_total_degree`, `all_multiindices_in_box`
  - `divides`, `is_constant`
  - `find_in_set`
  - `indices_in_box_with_bounded_degree`, `build_exponent_index_map`
  - `factorisations_assymetric`, `factorisations_fully_symmetric`, `factorisations_groupwise_symmetric`
  - `monomial_rank`, `num_multiindices_up_to`

#### 3. Polynomials.jl
- **External dependencies**: `LinearAlgebra`
- **Internal dependencies**: `..Multiindices` (uses `MultiindexSet`)
- **Exports**:
  - `AbstractPolynomial`, `DensePolynomial`
  - `polynomial_from_dict`, `polynomial_from_pairs`
  - `coeffs`, `multiindex_set`, `nvars`, `all_multiindices_up_to`
  - `coefficient`, `has_term`, `find_term`, `find_in_multiindex_set`
  - `zero`, `evaluate`, `extract_component`, `each_term`, `similar_poly`

#### 4. FullOrderModel.jl
- **External dependencies**: `LinearAlgebra`, `SparseArrays`
- **Internal dependencies**: None
- **Exports**:
  - `NDOrderModel`, `FirstOrderModel`
  - `MultilinearMap`
  - `linear_first_order_matrices`, `evaluate_nonlinear_terms!`

#### 5. Eigensolvers.jl
- **External dependencies**: `Arpack`, `LinearAlgebra`, `LinearMaps`, `SparseArrays`
- **Internal dependencies**: None
- **Exports**: `generalized_eigenpairs`

#### 6. Realification.jl
- **External dependencies**: `LinearAlgebra`
- **Internal dependencies**: `..Polynomials` (uses `AbstractPolynomial`, `DensePolynomial`)
- **Exports**:
  - `realify`, `compose_linear`, `realify_via_linear`
  - `_multinomial`, `_compositions`, `_reorder_canonical`, `_realify_term`

#### 7. Parametrisation.jl
- **Internal dependencies**: `..Polynomials` (uses `DensePolynomial`)
- **Exports**: `Parametrisation` (type alias for `DensePolynomial{NTuple{N,T}}`)

#### 8. MultilinearTerms.jl
- **Internal dependencies**:
  - `..Multiindices` (uses `MultiindexSet`, `factorisations_asymmetric`, `factorisations_fully_symmetric`, `factorisations_groupwise_symmetric`, `indices_in_box_with_bounded_degree`)
  - `..Polynomials` (uses `DensePolynomial`)
  - `..Parametrisation` (uses `Parametrisation`)
  - `..FullOrderModel` (uses `NDOrderModel`, `FirstOrderModel`, `MultilinearMap`)
- **Exports**: `compute_multilinear_terms`

#### 9. LowerOrderCouplings.jl
- **External dependencies**: `LinearAlgebra`
- **Internal dependencies**:
  - `.Multiindices` (uses `MultiindexSet`, `indices_in_box_with_bounded_degree`, `build_exponent_index_map`)
  - `..Polynomials` (uses `DensePolynomial`, `coeffs`, `multiindex_set`, `nvars`)
- **Exports**: `compute_lower_order_couplings`

## Public API Summary

| Module | Public Types | Key Functions |
|--------|-------------|---------------|
| `Multiindices` | `MultiindexSet` | Multiindex generation, binary search, factorization algorithms |
| `Polynomials` | `DensePolynomial` | Polynomial construction, evaluation, arithmetic |
| `FullOrderModel` | `NDOrderModel`, `FirstOrderModel`, `MultilinearMap` | Model representation, matrix conversion, term evaluation |
| `Eigensolvers` | - | `generalized_eigenpairs` |
| `Realification` | - | `realify`, `compose_linear`, `realify_via_linear` |
| `Parametrisation` | `Parametrisation` | Type alias for polynomial with tuple coefficients |
| `MultilinearTerms` | - | `compute_multilinear_terms` |
| `LowerOrderCouplings` | - | `compute_lower_order_couplings` |

## Data Flow

```
User Input
    │
    ▼
┌─────────────────┐
│  FullOrderModel │  ← Defines the N-th order ODE with linear/nonlinear terms
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  linear_first_order_matrices()                      │
│  Converts NDOrderModel → (A, B) matrices            │
└─────────────────────────────┬───────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Eigensolvers    │  ← Computes dominant eigenpairs
                    │ generalized_    │
                    │ eigenpairs()    │
                    └────────┬────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │  Parametrisation           │  ← DensePolynomial with NTuple coefficients
              │  (from subspace spanned    │
              │   by eigenvectors)        │
              └─────────────┬──────────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │  compute_multilinear_terms│  ← Evaluates nonlinear terms
              │  (in parametrised space)  │
              └────────────────────────────┘
```

## Notes

- The `deprecated_InputFullOrderModel/` directory contains legacy code and is not included in the main package.
- The `ParametrisationMethod/LinearOperator/` directory is currently empty.
- All polynomial operations use graded lexicographic (Grlex) ordering internally.
- The `MultilinearMap` type enforces that evaluation functions must be truly multilinear (linear in each argument independently).
