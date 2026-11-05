# Header map

Every production header lives under `include/tpic/` and is included by its path,
for example `#include <tpic/ecsim/maxwell/maxwell_device.hpp>`. A header named
`X_device.hpp` needs the CUDA or HIP compiler and includes
`lpfx/backend/cuda_hip/kernel.hpp` first; every other header is
host-includable. The include guard is the uppercase path. The layer each module
may include is in the [layer table](standards.md#layer-table); module ownership
and the runtime structure are in [architecture](architecture.md).
`include/lpfx/` is the bundled LPFx copy (see
[architecture](architecture.md#the-bundled-lpfx)) and is not mapped here.

| Module | Headers | Topics |
| --- | ---: | --- |
| `device` | 4 | TPIC block sizes, the x-fastest index decode, the cross-block reducer and the device validation flag. |
| `data` | 14 | Field and particle schemas and storage, the compact mass-matrix layout, output and checkpoint records. |
| `config` | 11 | Precision, species limits, input parsing, run options, domain, boundary conditions, physics, particle and output preparation. |
| `communication` | 10 | Field exchanges, the compact mass halo, particle migration and the transport routes. |
| `ecsim` | 37 | Deposition, field operators, GMRES, mass matrices, Maxwell and the field update, particle algorithms. |
| `initialization` | 18 | Simulation cases, analytic fields and host images, particle loading and centering. |
| `diagnostics` | 5 | Field and particle snapshots, output moments, scalar reductions, the Gauss-law residual. |
| `io` | 9 | Output and checkpoint engines, the asynchronous writer, the scalar CSV, restart reads and private serialization helpers. |
| `simulation` | 24 | Planning, execution, timing, state, transport, actions, reporting and driver. |

## device

TPIC's own device policy. The backend vocabulary comes from LPFx: the launch
statement, `LaunchGrid`, `EmptyWork`, `LaunchPreflight` and
`for_each_grid_index` from `lpfx/backend/cuda_hip/kernel.hpp`; `KernelLaunch`,
`default_block_threads` and `kernel_grid_block_limit` from
`lpfx/backend/launch.hpp`; the checked `create_event`, `record_event` and
`wait_event` from `lpfx/backend/error_reporting.hpp`; block reductions from
`lpfx/backend/cuda_hip/subgroup.hpp`.

| Header | Purpose |
| --- | --- |
| `device/block_reduction_device.hpp` | `reduce_block_partials_kernel`, the cross-block reducer shared by the scalar and Gauss diagnostics and GMRES with the LPFx summation tree. |
| `device/block_threads.hpp` | `particle_block_threads` and `mass_matrix_block_threads`, the block sizes of the kernel families that do not use the LPFx default; host-includable. |
| `device/device_validation.hpp` | `DeviceValidation`: the device failure flag, cleared and captured on the producing stream and checked on the host. |
| `device/thread_indexing.hpp` | `decode_x_fastest`, the x-fastest index decode; host-callable. |

## data

| Header | Purpose |
| --- | --- |
| `data/fields/field_schema.hpp` | `FieldSchema<Precision>`: the field schemas of nodal E, cell-centred B, node B and the current. |
| `data/fields/field_storage.hpp` | The device owners of the field schemas (`ElectricOwner`, `MagneticCellOwner`, `MagneticNodeOwner`, `CurrentOwner`, `OutputMomentOwner`, `GaussCellOwner`), `DeviceFieldImage`, `upload_field` and `make_zero_field`. |
| `data/fields/grid_shape.hpp` | `GridShape` and `GridHalo`, the extents passed by value to kernels. |
| `data/mass_matrix/compact_mass_matrix_device.hpp` | `CompactMassMatrix`: the persistent mass coefficients without the volume factor. |
| `data/mass_matrix/mass_matrix_layout.hpp` | `MassMatrixLayout`: canonical spatial-pair order of the stored mass stencil. |
| `data/output/checkpoint.hpp` | `CheckpointHeader`, the checkpoint field values, and `PreparedRestart`: the pieces and records a rank restarts from. |
| `data/output/output_moment_quantities.hpp` | Output-moment quantities, components and selection bits. |
| `data/output/output_moment_schema.hpp` | `OutputMomentSchema`: the physical output moments, accumulated in FP64. |
| `data/output/output_snapshot.hpp` | Host snapshot records and views shared by capture and the writers: owned field arrays and particle records lent from the capture's pinned images. |
| `data/output/scalar_sample.hpp` | `ScalarDiagnosticSample` and the scalar positions in storage and column order. |
| `data/output/gauss_sample.hpp` | `GaussSample`, the Gauss slot order and `GaussFieldSchema`. |
| `data/particles/particle_cell.hpp` | `GridIndex` and `ParticleCellMap`: rank-local particle cell keys with their halo. |
| `data/particles/particle_fractions.hpp` | `ParticleFractions`: load of a record's cell-relative fractions and the canonical-range predicate. |
| `data/particles/particle_schema.hpp` | `ParticleSchema<Precision>` and the rebuild policy of the local cell and fractions. |

## config

| Header | Purpose |
| --- | --- |
| `config/boundary_conditions.hpp` | The one definition of the boundary conditions: `FieldBoundary`, `ParticleBoundary`, one rule per kind (`field_boundary_rules`, `particle_boundary_rules`) and how to add a kind. |
| `config/detail/errors.hpp` | `Errors`: the `TPIC <topic>: ` exceptions of one config topic. |
| `config/domain.hpp` | `PreparedDomain`, `prepare_domain` and the canonical partition check. |
| `config/limits.hpp` | Real-species capacity (`max_real_species`, from `TPIC_MAX_SPECIES`) and `require_species_count`. |
| `config/numerics_options.hpp` | `ParticleOrder`, `MassDepositionKernel`, `MassApplyMode` with names, parsers and the default order. |
| `config/output.hpp` | `OutputEngineKind`, `PreparedOutput` and `prepare_output`. |
| `config/output_selection.hpp` | `PreparedOutputSelection` and the field snapshot slots. |
| `config/particle_plan.hpp` | `ParticlePlan`: per-species counts, capacities and the migration bypass. |
| `config/physics.hpp` | `PreparedPhysics`: field and particle boundary kinds per face. |
| `config/precision.hpp` | The storage and compute precision structs, the pair `Precision<Storage, Compute>`, `convert`, `PrecisionSelection` with its parsers and metadata, and `dispatch_precision`. |
| `config/simulation_parameters.hpp` | `SimulationParameters` and the input-deck parser. |

## communication

| Header | Purpose |
| --- | --- |
| `communication/fields/field_exchange.hpp` | `FieldExchangeEpoch` and `complete_field_exchange`. |
| `communication/mass_halo/mass_halo_device.hpp` | `MassHaloStorage` (pack and unpack kernels, `enqueue_pack`, `enqueue_unpack`). |
| `communication/mass_halo/mass_halo_plan.hpp` | `MassHaloPlan::prepare` over the two `MassHaloPhasePlan`s, `MassHaloDirection`, `MassHaloSide`, `mass_halo_stencil_mask`, `HaloProgress` and `MassHaloLaunch`. |
| `communication/particles/initial_particle_classification_device.hpp` | Startup ownership pass: `enqueue_initial_particle_classification` and `enqueue_local_initial_particle_boundaries`. |
| `communication/particles/local_particle_destination_sink_device.hpp` | `LocalParticleDestinationSink`: destination sink of a whole-domain rank. |
| `communication/particles/particle_boundaries.hpp` | Particle boundary handling, destination flags and `ParticleDestinationFlagDecoder`. |
| `communication/particles/particle_geometry.hpp` | `ParticleMigrationGeometry` and its preparation. |
| `communication/particles/particle_migration_policy.hpp` | `particle_migration_policy`: the descriptor's boundary and overlap identities and `particle_migration_hops` (from `TPIC_PARTICLE_MIGRATION_HOPS`). |
| `communication/particles/particle_wire.hpp` | Particle wire codecs, the final-arrival rebuilders and the wire contract `ParticleMigrationWire`. |
| `communication/transport_routes.hpp` | `DeviceRoute` and `GmresRoute` for the MPI, NCCL and RCCL builds. |

## ecsim

| Header | Purpose |
| --- | --- |
| `ecsim/deposition/current_assembly_device.hpp` | `CurrentAssembly`: shared-node sum, dual-volume scale, smoothing halo and Neumann ghosts of the current. |
| `ecsim/deposition/current_deposition_device.hpp` | Current kernels and `enqueue_unsorted_current`, `enqueue_sorted_current`, `enqueue_resident_current`. |
| `ecsim/deposition/deposition_admission_device.hpp` | `ConstParticleReadView` and `check_completed_particle_range`. |
| `ecsim/deposition/moment_field_device.hpp` | `MomentFieldPlan` and the clear, dual-volume and Neumann-ghost kernels of a moment field. |
| `ecsim/fields/cell_to_node.hpp` | Cell-to-node average arithmetic and its preparation. |
| `ecsim/fields/cell_to_node_device.hpp` | `enqueue_magnetic_cell_to_node`. |
| `ecsim/fields/field_boundary.hpp` | `FieldBoundaryPlan` and `FieldBoundaryPlans`: the wall ghosts and the constrained rows of one domain. |
| `ecsim/fields/field_boundary_device.hpp` | Wall ghost and constrained electric kernels with their enqueue functions. |
| `ecsim/fields/field_checks.hpp` | Owner shape and location checks shared by the field operators. |
| `ecsim/fields/field_vector_device.hpp` | `field_vector_shape`, `ScaledField`, linear combination and copy. |
| `ecsim/fields/node_cell_curl.hpp` | Node-cell curl spacing and preparation. |
| `ecsim/fields/node_cell_curl_device.hpp` | `enqueue_node_cell_curl`. |
| `ecsim/fields/node_smoothing.hpp` | `SmoothingWeights` and their preparation. |
| `ecsim/fields/node_smoothing_device.hpp` | `enqueue_node_smoothing`. |
| `ecsim/fields/stencil_region.hpp` | `StencilRegion` and the region geometry of the split Maxwell apply. |
| `ecsim/fields/wall_faces.hpp` | `WallFaces`: relative inverse nodal control volumes at walls. |
| `ecsim/gmres/gmres.hpp` | GMRES sizes shared with the memory estimate and `GmresLaunch`. |
| `ecsim/gmres/gmres_device.hpp` | `Gmres` and its kernels. |
| `ecsim/mass_matrix/mass_matrix_apply_device.hpp` | `enqueue_mass_matrix_apply`. |
| `ecsim/mass_matrix/mass_matrix_deposition_device.hpp` | `enqueue_clear_mass_matrix`, `enqueue_unsorted_mass_matrix`, `enqueue_sorted_mass_matrix`, `enqueue_resident_mass_matrix` and `matrix_core_mass_deposition_available`. |
| `ecsim/mass_matrix/mass_matrix_mfma_device.hpp` | HIP FP64 MFMA4 sorted mass kernel and `MassMfma4Launch`. |
| `ecsim/mass_matrix/mass_matrix_mfma_layout.hpp` | `MassMfma4Layout` and the supported MFMA4 architectures. |
| `ecsim/mass_matrix/mass_matrix_mma_device.hpp` | CUDA FP64 MMA sorted mass kernel. |
| `ecsim/mass_matrix/mass_matrix_scalar_device.hpp` | Scalar unsorted and sorted mass kernels. |
| `ecsim/mass_matrix/mass_matrix_sorted_cell_device.hpp` | Subgroup lane maps of the sorted mass kernels. |
| `ecsim/mass_matrix/mass_matrix_stencil.hpp` | `MassPairStencils`, `MassSubgroupLayout` and `MassLaneStencilMap`. |
| `ecsim/maxwell/field_update_device.hpp` | `enqueue_next_fields` and the Faraday update kernel. |
| `ecsim/maxwell/maxwell.hpp` | `SolverFieldSchema`, `MaxwellParameters`, `FieldOperatorPlans` and `prepare_field_operator_plans`. |
| `ecsim/maxwell/maxwell_apply_policy.hpp` | `MaxwellApplySchedule` and `resolve_maxwell_apply`. |
| `ecsim/maxwell/maxwell_device.hpp` | `MaxwellOperator` and the solver owner aliases `SolverNodeOwner`, `SolverCellOwner`. |
| `ecsim/particles/cic.hpp` | `CicWeights` and the cloud-in-cell corner order. |
| `ecsim/particles/particle_mover_device.hpp` | Mover kernels, `enqueue_migrating_particle_mover` and `enqueue_local_particle_mover`. |
| `ecsim/particles/particle_mover_parameters.hpp` | `ParticleMoverParameters`, prepared once per species. |
| `ecsim/particles/particle_response.hpp` | `ParticleResponse`: the 3x3 response tensor. |
| `ecsim/particles/particle_sample_device.hpp` | `ParticleSample` with the `DecodeKey` and `KnownBin` cell sources. |
| `ecsim/particles/particle_sort_device.hpp` | `ParticleSortOwner`, `FullParticleSort` and `PartialParticleSort`. |
| `ecsim/particles/sorted_cell_device.hpp` | `SortedCellLaunch`, `ParticleBins` and the cached cell values of the sorted kernels. |

## initialization

| Header | Purpose |
| --- | --- |
| `initialization/cases/case_interface.hpp` | The standard shape of a simulation case, with `FieldVector`, `ParticleProfileSample` and `UniformProfile`. |
| `initialization/cases/initial_case.hpp` | `InitialCase`, the one list of cases; `prepare_initial_case`, `initial_electric`, `initial_magnetic` and the admission checks shared by the cases. |
| `initialization/cases/<name>.hpp` | One case each: `UniformCase`, `HarrisSheetCase`, `ForceFreeCase`, `WeibelCase`, `WhistlerCase`, `TaylorGreenVortexCase`, `TwoStreamCase`, `FluxRopeCase`, `GaussianExpansionCase`, `WhistlerTurbulenceCase`, with the device profile of its particle loading. |
| `initialization/fields/field_image.hpp` | `HostFieldImage` and `make_host_field_image`. |
| `initialization/particles/particle_centering_device.hpp` | `VelocityCentering` and its kernel. |
| `initialization/particles/particle_load_plan.hpp` | `ParticleLoadPlan` and `velocity_centering_reductions`. |
| `initialization/particles/particle_support.hpp` | `ParticleSupport`, the region a profile loads, and `visit_support_rows`, the host's rows of subcells of a bounded one. |
| `initialization/particles/particle_loading_device.hpp` | `enqueue_initial_particle_load` and `make_initial_particles`. |
| `initialization/prepared_initialization.hpp` | `prepare_bounded_particle_plan` and `prepare_particle_loads`. |

## diagnostics

| Header | Purpose |
| --- | --- |
| `diagnostics/async_field_snapshot_device.hpp` | `AsyncFieldSnapshot`: packing and asynchronous download of selected fields. |
| `diagnostics/async_particle_snapshot_device.hpp` | `AsyncParticleSnapshot`: particle records copied as stored into one pinned image per species, lent to the writer. |
| `diagnostics/output_moment_deposition_device.hpp` | `enqueue_deposit_output_moments`. |
| `diagnostics/scalar_diagnostics_device.hpp` | `ScalarDiagnostics`: field energy and species scalars. |
| `diagnostics/gauss_residual_device.hpp` | `GaussResidual`: the Gauss-law residual of the committed states. |

## io

| Header | Purpose |
| --- | --- |
| `io/async_output_writer.hpp` | `AsyncOutputWriter`: the persistent disk writer worker. |
| `io/checkpoint.hpp` | The checkpoint engine factory, checkpoint discovery and set-aside, and the restart reads of fields and particles. |
| `io/detail/checkpoint_engine.hpp` | The checkpoint layout, `CheckpointEngine` and `CheckpointReader`, shared by ADIOS2 and HDF5. |
| `io/detail/dataset_engine.hpp` | `DatasetOutputEngine` and `StagedOutput`, shared by ADIOS2 and HDF5. |
| `io/detail/output_file.hpp` | Output file names and `OutputFile`. |
| `io/detail/output_layout.hpp` | The particle array descriptors and the conversion of stored records to output values. |
| `io/detail/vtk_output.hpp` | VTK XML helpers. |
| `io/diagnostics_writer.hpp` | `DiagnosticsWriter` and `GaussWriter`: the rank-0 scalar CSV files, sharing `detail::CycleCsvFile`. |
| `io/output_engine.hpp` | `OutputEngine` interface and its factories. |

## simulation

| Header | Purpose |
| --- | --- |
| `simulation/planning/run_plan.hpp` | `InputOptions`, `RunMode` and `RuntimeDebugOptions`; `PreparedInput`, `prepare_input` and `report_particle_bytes`; `RunPlan`, the immutable owner of host preparation. |
| `simulation/execution/active_backend_device.hpp` | The aliases `Backend`, `Executor` and `Event` of the build's device backend. |
| `simulation/execution/deposition_ledger.hpp` | `DepositionLedger`: exact-once deposition order. |
| `simulation/execution/execution_layout.hpp` | `ExecutionLayout`: every cardinality of a run; `Epoch`, `MomentQuantity`, and the deposition records. |
| `simulation/execution/execution_streams.hpp` | `ExecutionStreams` and `StreamExecutors`. |
| `simulation/timing/stage_timing.hpp` | `StageTiming`: host and GPU timing scopes. |
| `simulation/timing/timing_regions.hpp` | `RunningTimingStats`, `TimingRegion` and its table: names, event slots and print order. |
| `simulation/state/species_device.hpp` | `Species<Precision, Partial>`, the species receipt helpers and `visit_species`: run-time species index to compile-time index. |
| `simulation/transport/transport_registry_device.hpp` | `TransportRegistry` and the species pack. |
| `simulation/actions/deposition_actions_device.hpp` | `DepositionActions`. |
| `simulation/actions/diagnostics_actions_device.hpp` | `DiagnosticsActions`. |
| `simulation/actions/field_advance_actions_device.hpp` | `FieldAdvanceActions`. |
| `simulation/actions/field_solve_actions_device.hpp` | `FieldSolveActions`. |
| `simulation/actions/local_species_actions_device.hpp` | `LocalSpeciesActions`: migration, route, ordering, centering and the startup steps of a species. |
| `simulation/actions/magnetic_startup_device.hpp` | `complete_initial_magnetic_field`. |
| `simulation/actions/moment_submission_error.hpp` | `check_moment_submission`. |
| `simulation/actions/mover_actions_device.hpp` | `MoverActions`. |
| `simulation/actions/output_actions_device.hpp` | `OutputActions`: snapshots and output moments. |
| `simulation/reporting/memory_estimator.hpp` | `CheckedSize` and the sizing functions of the estimate. |
| `simulation/reporting/run_reporter.hpp` | `RunReporter`: rank-0 console lines. |
| `simulation/reporting/startup_report.hpp` | `RunResult`, `MemoryEstimate`, `RankMemoryReport`, the startup and completion reports and `print_error`, the one error line every failure path prints. |
| `simulation/driver/run_distributed.hpp` | `run_distributed` and the collective reports of the driver. |
| `simulation/driver/pipeline.hpp` | `Pipeline`: the runtime of a run; host-compilable. |
| `simulation/driver/simulation_device.hpp` | `Simulation<Precision, Partial, Kernel>`, the physics owner, and `run_simulation<Precision>`: the order and kernel dispatch to it. |

## Sources

| Module | Sources under `src/` |
| --- | --- |
| `app` | `main.cpp` |
| `config` | `domain.cpp`, `output.cpp`, `output_selection.cpp`, `particle_plan.cpp`, `physics.cpp`, `simulation_parameters.cpp` |
| `initialization` | `initial_case.cpp`, `prepared_initialization.cpp`, and one file per case except `uniform` in `cases/` |
| `io` | `async_output_writer.cpp`, `checkpoint.cpp`, `dataset_engine.cpp`, `diagnostics_writer.cpp`, `output_engine.cpp` and the engines `vtk_field_engine.cpp`, `vtk_particle_engine.cpp`, `adios2_engine.cpp`, `hdf5_engine.cpp`; the last two also hold their checkpoint engines and readers, and each is compiled only when its backend is enabled (`dataset_engine.cpp` when either is) |
| `simulation` | Host: `moment_submission_error.cpp`, `run_plan.cpp`, `run_reporter.cpp`, `startup_report.cpp`, `timing_regions.cpp`. Device: `run_distributed.cpp` and `simulation_dispatch.cpp`, the latter compiled once per precision pair |
