# Agents.md

This repository is the **integration layer** between an existing FEEC library and an existing GMRF library. The initial implementation will use the **explicit sparse-matrix route** (assemble Jacobians / Hessians and solve via sparse linear algebra).

This document defines the recommended **multi-repo structure**, crate boundaries, and dependency rules so development stays acyclic and each library keeps a clear responsibility.

---

## 1) Repositories and responsibilities

### Repo A: `feec` (existing)

**Responsibility:** discretization and PDE/FEEC algebra.

**Owns:**

- function spaces / dof maps / mesh topology
- assembly of FEEC matrices (mass, stiffness, Hodge stars, incidence operators, mixed blocks)
- assembly of nonlinear PDE residual vectors `r(x)`
- assembly of the Jacobian `J(x)` of the residual (explicit sparse matrix for the current explicit-matrix plan)

**Does not own:**

- Gauss–Newton / Laplace / MAP orchestration
- generic Gaussian conditioning / precision bookkeeping (beyond providing FEEC matrices)

**Exports (for integration):**

- assembled FEEC matrices used as prior precision components
- assembled residual `r(x)` and Jacobian `J(x)` for “PDE observation” likelihood terms

---

### Repo B: `gmrf` (existing)

**Responsibility:** Gaussian / precision algebra and sparse linear solves.

**Owns:**

- representations of precision matrices / Gaussian models
- sparse solves and (optionally) sparse factorizations
- logdet / quadratic forms / sampling (as available)

**Does not own:**

- FEEC details (spaces, forms, quadrature, boundary assembly)
- nonlinear modeling (residual definitions, Jacobian formation)

**Exports (for integration):**

- a small set of supported sparse matrix backends/types (or wrapper types)
- `solve_spd(Q, b)` and/or `factorize_spd(Q)` APIs
- (optional) utilities for structured precisions (e.g., block tridiagonal)

---

### Repo C: `feec-gmrf-integration` (this repo)

**Responsibility:** statistical model composition and inference algorithms (Gauss–Newton / Laplace).

**Owns:**

- composition of priors + likelihood terms into a single objective
- Gauss–Newton iteration, damping/line-search/trust region policy
- Laplace approximation outputs (MAP, approximate posterior precision)
- spatiotemporal prior _construction_ (turn FEEC matrices + time discretization into a Gaussian precision)

**Does not own:**

- FEEC assembly internals
- low-level sparse solver implementations

---

## 2) Avoiding cyclic dependencies (required)

To allow **FEEC types to implement integration-facing traits** without a `feec <-> integration` cycle, shared traits/types must not live in the integration crate directly.

Use one of the following patterns:

### Pattern 1 (recommended): a small shared crate `feg-core` in this repo

Create a crate containing only shared traits and shared sparse types.

- `feec` depends on `feg-core` (optional feature)
- `gmrf` depends on `feg-core` (optional feature)
- integration depends on `feg-core`, `feec`, and `gmrf`

This avoids cycles and keeps public contracts stable.

> If `feec` and `gmrf` live in separate repos, they can depend on `feg-core` via a git dependency pointing at this repo + `path = "crates/feg-core"`, or `feg-core` can be split into its own small repo later if needed.

### Pattern 2: publish `feg-core` as its own repo/crate

Same idea, but `feg-core` is independent and versioned separately. Use this if cross-repo git-path deps are undesirable.

---

## 3) Explicit matrix route: where assembly and solves happen

### Assembly ownership

- `feec` assembles:
  - residual vector `r(x)`
  - Jacobian matrix `J(x)` (sparse)
  - FEEC prior matrices (mass/stiffness/Hodge/etc.) used to form prior precision `Q_prior`

### MAP / Gauss–Newton step ownership

- integration forms the Gauss–Newton quadratic (example):
  - `H = Q_prior + J(x)^T * R^{-1} * J(x)` (sparse SPD, typically)
  - `g = Q_prior*(x - mu) + J(x)^T * R^{-1} * r(x)`
- integration calls `gmrf` to solve:
  - `H * dx = -g`
- integration updates `x <- x + alpha * dx` (damping / line search owned here)

### Resulting boundary

- `feec` produces explicit matrices and residuals
- `gmrf` solves explicit SPD systems and provides Gaussian algebra
- integration owns the nonlinear loop and how terms are combined

---

## 4) Spatiotemporal prior: where it should live

### Construction belongs in integration

Construction depends on:

- chosen time discretization (implicit Euler / Crank–Nicolson / etc.)
- chosen state definition per time slice (which FEEC space(s))
- chosen process noise model and scaling

These are modeling choices, so they belong in **integration**.

### Representation/solves belong in `gmrf`

If the prior yields a structured precision (e.g., **block tridiagonal**), the _data structure_ and _fast solver_ should live in `gmrf` (or in a dedicated submodule of `gmrf`). Integration should only instantiate it.

**Rule:**

- integration computes blocks (from FEEC matrices + time grid)
- gmrf stores/solves the resulting structured Gaussian

---

## 5) Recommended crate layout inside this integration repo

This repo should be a Cargo workspace:

```
feec-gmrf-integration/
  Cargo.toml                 # workspace
  Agents.md
  crates/
    feg-core/
      Cargo.toml
      src/
        lib.rs               # shared traits + shared sparse matrix wrapper types
        sparse.rs            # wrappers / conversions for the chosen backend
    feg-infer/
      Cargo.toml
      src/
        lib.rs
        gn/                  # Gauss–Newton loop + damping policies
        laplace/             # Laplace outputs (MAP, approx precision access)
        model/               # likelihood/prior composition
        prior/
          spacetime.rs       # implicit Euler, block-tridiagonal construction
        util/
    feg-examples/            # optional: runnable examples / benches
      Cargo.toml
      src/
```

### `feg-core` contents (explicit matrix plan)

`feg-core` should define:

- `ResidualModel` trait:
  - `residual(x) -> r`
  - `jacobian(x) -> J` (explicit sparse matrix)

- a single “blessed sparse matrix wrapper type” `SparseMat` that:
  - is owned by `feg-core` (newtype wrapper)
  - can wrap the selected backend (e.g., `sprs`, `faer`, etc.)
  - avoids orphan-rule issues when implementing shared traits

- minimal traits needed by `gmrf` (e.g., `SpdSolve` inputs), but keep it small.

### `feg-infer` contents

`feg-infer` orchestrates:

- MAP solve loop (Gauss–Newton) using explicit `J`/`H`
- Laplace approximation (store MAP, store/factorize `H` if needed)
- spatiotemporal prior construction:
  - consumes FEEC matrices (`M`, `K`, mixed blocks)
  - produces a `gmrf` precision object (dense blocks or sparse blocks, depending on implementation)

---

## 6) Feature flags (suggested)

The goal is to keep dependencies optional and explicit.

### `feg-core`

- `sprs` or `faer` (pick one backend initially; add the other later if needed)

### `feec`

- `feg-core` (enables impls of `feg-core` traits)
- `explicit-jacobian` (enables `jacobian(x)` assembly API)

### `gmrf`

- `explicit` (accept explicit sparse matrix wrapper type from `feg-core`)
- `direct` (enables factorization-based solves)
- `logdet` (optional)

### `feg-infer`

- `gn` (Gauss–Newton)
- `laplace` (Laplace outputs; depends on `gmrf/direct` if using exact logdet/factors)
- `spacetime` (spatiotemporal prior construction utilities)

---

## 7) Dependency rules (must follow)

1. `feec` must **not** depend on `gmrf` or `feg-infer`.
2. `gmrf` must **not** depend on `feec` or `feg-infer`.
3. `feg-infer` may depend on both `feec` and `gmrf`.
4. Shared traits/types live in `feg-core` (or a separate shared repo), not in `feg-infer`.

---

## 8) Development checklist for agents

When adding a new feature, first decide which category it is:

- **Discretization / PDE math / assembly** → `feec`
- **Gaussian algebra / solves / factorization / structured precision algorithms** → `gmrf`
- **Model composition / inference loop / Laplace / time discretization choices** → `feg-infer`
- **Shared traits/types required by both sides** → `feg-core` (keep minimal)

If a change would introduce a cyclic dependency, refactor by moving the shared interface into `feg-core`.
