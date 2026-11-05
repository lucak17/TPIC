# TPIC documentation

## User

| Guide | Contents |
| --- | --- |
| [Build and run](user/build_and_run.md) | Prerequisites, presets, CMake options, device architectures, command-line modes and options, exit codes, rank/GPU placement, first run, console report, environment variables |
| [Inputs](user/inputs.md) | Deck syntax, units and normalization, decomposition, walls and smoothing, particle loading, `QuietStart`, seeds, `CenterInitialVelocity` |
| [Input keys](user/input_keys.md) | Every input key with its type, default and meaning |
| [Cases](user/cases.md) | The shipped cases: decks, dimension, size, ranks, needed outputs, analyzer, paper, admission |
| [Gaussian expansion](user/gaussian_expansion.md) | `GaussianExpansion` case: setup, production, quick and octant runs, diagnostics D1-D14, acceptance checks |
| [Whistler turbulence](user/whistler_turbulence.md) | `WhistlerTurbulence` case: units, implicit settings, the seed, validation runs V0-V4, analysis |
| [Output](user/output.md) | Output files, cadences, tags, engines, layouts, ParaView/Fides/XDMF, checkpoints and restart |
| [Analysis](user/analysis.md) | Post-processing tools and the analyzer of each case |
| [Gauss's law](user/gauss_law.md) | What the Gauss-law check measures, why it is not zero, how to monitor it |
| [Support](user/support.md) | Supported scope, known limits, tested platforms, matrix-core kernels, known platform issues |
| [Scaling](../performance/scaling/README.md) | Strong- and weak-scaling campaigns on clusters |

## Developer

| Guide | Contents |
| --- | --- |
| [Architecture](developer/architecture.md) | Modules, layers, ownership, runtime, pipeline, streams, events, workers, communication, the bundled LPFx |
| [Headers](developer/headers.md) | Every header and source file |
| [Kernels](developer/kernels.md) | Every device kernel, its launch shape and owner |
| [Standards](developer/standards.md) | Engineering rules R1-R7 and the checks that enforce them |
| [Change review](developer/change_review.md) | Change classes and how a change is qualified |
| [Numerical reference](developer/numerical_reference.md) | Equations, staggering, stencils, rounding, solver, boundaries, startup, matrix-core kernels |
| [Output architecture](developer/output_architecture.md) | Output, writer and checkpoint internals |
| [Gauss diagnostic](developer/gauss_diagnostic.md) | Gauss-law residual diagnostic internals |
| [Testing](developer/testing.md) | Test suites, golden tables, integration tiers, tolerances |
| [Performance](developer/performance.md) | Timing and profiling tools |
