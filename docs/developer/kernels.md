# Kernel inventory

Every device kernel (`__global__` function) of `include/tpic/`, with the
function that launches it, its launch shape and work count, its iteration model
and whether it reports through a device validation flag. There are 42 kernels
in 23 headers; 17 take an `unsigned* invalid` flag. Kernel bodies, launch
configurations and shared-memory layouts are part of the performance contract
of the [engineering standards](standards.md): a change to them is a class K or
L change of the [change review](change_review.md). The LPFx kernels of
`include/lpfx/` are not listed.

Every kernel lives in `tpic::<module>::kernels` and is launched only through
LPFx's `<lpfx/backend/cuda_hip/kernel.hpp>`: `lpfx::device::launch` returns an
`lpfx::ErrorCode` from an enqueue function, `lpfx::device::launch_or_throw` throws
from an owner class whose host code reports by exception (`Gmres`,
`ScalarDiagnostics`, `GaussResidual`, `AsyncFieldSnapshot`, `MassHaloStorage`,
`VelocityCentering`). Both submit with
no dynamic shared memory on the executor's stream and add no synchronisation;
`launch` adds one only when a submission diagnostic is passed, and only
`launch_or_throw` takes an explicit `lpfx::device::LaunchGrid`. Tests may launch a module
kernel from `<module>::kernels` with a geometry of their own; a test that uses
the kernel's own launch shape calls the enqueue function instead.

## Launch shapes

A shape is `lpfx::device::KernelLaunch<Threads, WorkPerBlock>` from
`<lpfx/backend/launch.hpp>`: `Shape::blocks(work)` is the ceiling of
`work / WorkPerBlock`, capped at `lpfx::device::kernel_grid_block_limit` (65535); the
kernel covers the rest of its work with `lpfx::device::for_each_grid_index`.
The kernel's `__launch_bounds__` names a constant equal to the block size of its
launch; TPIC's own block sizes are in `device/block_threads.hpp`, except
`mass_mfma4_threads`, which follows the MFMA4 layout.

| Shape | Threads | Work per block | Used by |
| --- | ---: | --- | --- |
| `lpfx::device::KernelLaunch<>` | `lpfx::device::default_block_threads` (256) | 256 items | Field, solver, diagnostics, startup and unsorted particle kernels |
| `lpfx::device::KernelLaunch<device::particle_block_threads>` | 128 | 128 particles | Unsorted mass-matrix deposition |
| `lpfx::device::KernelLaunch<device::mass_matrix_block_threads>` | 128 | 128 items | Mass-matrix apply |
| `communication::MassHaloLaunch` | `mass_matrix_block_threads` (128) | 128 entries | Mass-halo pack and unpack |
| `ecsim::SortedCellLaunch` | `sorted_cell_threads` (128) | `sorted_cell_max_warps` cell bins | Sorted current, mass-matrix (CUDA, and HIP scalar) and mover kernels: one native subgroup per cell bin |
| `ecsim::MassMfma4Launch` | `mass_mfma4_threads` (four wave64) | four cell bins | HIP matrix-core mass-matrix deposition |
| `ecsim::GmresLaunch` | `gmres_reduction_threads` (256) | 256 items | GMRES vector kernels |
| `detail::ScalarLaunch` (`diagnostics/scalar_diagnostics_device.hpp`) | 256 | 256 items | Scalar diagnostics and the Gauss residual |

`lpfx::device::EmptyWork::skip` returns success without a launch when the work count
is zero; `lpfx::device::EmptyWork::launch` launches the computed grid as written.
`lpfx::device::LaunchPreflight::admitted` marks a launch whose enqueue function
already consumed the launch-error slot with
`lpfx::device::begin_device_submission`. An explicit `lpfx::device::LaunchGrid`
gives the grid of a launch whose geometry is not `Shape::blocks(work)`.

## communication

| Kernel | Header | Launched by | Shape and work | Iteration | Flag |
| --- | --- | --- | --- | --- | --- |
| `mass_halo_pack_kernel<Scalar>` | `communication/mass_halo/mass_halo_device.hpp` | `MassHaloStorage::enqueue_pack` (both phases) | `LaunchGrid{phase.pack_blocks(), MassHaloLaunch::threads}`, one launch per phase over every direction | The grid concatenates the blocks of every direction's send side; each block finds its direction from the `first_block` offsets with one warp ballot, then a grid-stride loop over that side's entries | no |
| `mass_halo_unpack_add_kernel<Scalar>` | same | `MassHaloStorage::enqueue_unpack` (additive phase) | `LaunchGrid{phase.unpack_blocks(), MassHaloLaunch::threads}` | As the pack kernel over the receive sides; one thread per entry adds every direction that carries it | no |
| `mass_halo_unpack_copy_kernel<Scalar>` | same | `MassHaloStorage::enqueue_unpack` (ghost phase) | `LaunchGrid{phase.unpack_blocks(), MassHaloLaunch::threads}` | As the pack kernel over the receive sides | no |
| `initial_particle_classification_kernel` | `communication/particles/initial_particle_classification_device.hpp` | `enqueue_initial_particle_classification`, `enqueue_local_initial_particle_boundaries` | `KernelLaunch<>` over the stored records, `EmptyWork::skip` | `lpfx::device::for_each_grid_index` over the producer's records | through `LocalParticleDestinationSink` in the local pass |

## device

| Kernel | Header | Launched by | Shape and work | Iteration | Flag |
| --- | --- | --- | --- | --- | --- |
| `reduce_block_partials_kernel<T>` (`tpic::device::kernels`) | `device/block_reduction_device.hpp` | `ScalarDiagnostics` (`enqueue_scalar_reduction`), `GaussResidual` (`enqueue_reduction`), `Gmres` (`enqueue_batch_partial_reduction`) | `LaunchGrid{rows, threads}`: one block per output row | Thread-stride sum of one row's block partials with the LPFx block-reduction tree; rows at or above the filled count store zero | no |

## diagnostics

| Kernel | Header | Launched by | Shape and work | Iteration | Flag |
| --- | --- | --- | --- | --- | --- |
| `pack_field_snapshot_kernel` | `diagnostics/async_field_snapshot_device.hpp` | `AsyncFieldSnapshot` (`pack_to`) | `KernelLaunch<>` over the field samples | `lpfx::device::for_each_grid_index` | no |
| `accumulate_field_snapshot_kernel` | same | `AsyncFieldSnapshot` (`accumulate_to`), CUDA | `KernelLaunch<>` over the field samples | `lpfx::device::for_each_grid_index`; `__dmul_rn` rounds the scaled value before the addition | no |
| `add_packed_field_snapshot_kernel` | same | `AsyncFieldSnapshot` (`add_values`), HIP second launch | `KernelLaunch<>` over the packed values | Grid-stride | no |
| `deposit_output_moments_kernel<Precision, Pressure, ...>` | `diagnostics/output_moment_deposition_device.hpp` | `enqueue_deposit_output_moments` (also the Gauss charge, a rho-only owner) | `KernelLaunch<>` over the species' particles, `EmptyWork::skip` | `lpfx::device::for_each_grid_index` | yes |
| `field_energy_partials_kernel` | `diagnostics/scalar_diagnostics_device.hpp` | `ScalarDiagnostics::enqueue_field_energy` | `LaunchGrid{blocks, ScalarLaunch::threads}`, then `reduce_block_partials_kernel` | `lpfx::device::for_each_grid_index` over cells and nodes; one partial per component and block | no |
| `species_scalar_partials_kernel` | same | `ScalarDiagnostics::enqueue_species_scalars` | `LaunchGrid{blocks, ScalarLaunch::threads}`, then `reduce_block_partials_kernel` | `lpfx::device::for_each_grid_index`; one partial per component and block | no |
| `gauss_field_partials_kernel<Begin, Finish, ...>` | `diagnostics/gauss_residual_device.hpp` | `GaussResidual::enqueue_field` | `LaunchGrid{blocks, ScalarLaunch::threads}`, then `reduce_block_partials_kernel` when finishing | `lpfx::device::for_each_grid_index` over local cells; finishing reads the stash before beginning stores it | no |
| `gauss_charge_partials_kernel` | same | `GaussResidual::complete_charge` | `LaunchGrid{blocks, ScalarLaunch::threads}`, then `reduce_block_partials_kernel` | `lpfx::device::for_each_grid_index` over local cells; one partial per block | no |

## ecsim/deposition

| Kernel | Header | Launched by | Shape and work | Iteration | Flag |
| --- | --- | --- | --- | --- | --- |
| `unsorted_current_kernel` | `ecsim/deposition/current_deposition_device.hpp` | `enqueue_unsorted_current` | `KernelLaunch<>` over the particle range, `EmptyWork::skip` | `lpfx::device::for_each_grid_index`, one particle per index | yes |
| `sorted_current_kernel` | same | `detail::launch_sorted_current`, from `enqueue_sorted_current` and `enqueue_resident_current` | `SortedCellLaunch` over the cell bins | One subgroup per cell bin, stride over bins; shared magnetic cache and a padded contribution tile of `sorted_cell_threads + lanes_per_corner` columns | yes |
| `scale_moment_dual_volume_kernel` | `ecsim/deposition/moment_field_device.hpp` | `enqueue_scale_moment_dual_volume` | `KernelLaunch<>` over the shell of the interior-node box; no launch without a wall | `lpfx::device::for_each_grid_index`, each node once | no |
| `fill_moment_neumann_ghost_kernel` | same | `enqueue_fill_moment_neumann_ghost` | `KernelLaunch<>` over the ghost shell; no launch without a wall | `lpfx::device::for_each_grid_index` | no |
| `clear_moment_field_kernel` | same | `enqueue_clear_moment_field` | `KernelLaunch<>` over every node, ghosts included | `lpfx::device::for_each_grid_index` | no |

## ecsim/fields

| Kernel | Header | Launched by | Shape and work | Iteration | Flag |
| --- | --- | --- | --- | --- | --- |
| `magnetic_cell_to_node_kernel` | `ecsim/fields/cell_to_node_device.hpp` | `enqueue_magnetic_cell_to_node` | `KernelLaunch<>` over the interior nodes | `lpfx::device::for_each_grid_index` | yes |
| `wall_ghost_kernel<Axis, Node, ...>` | `ecsim/fields/field_boundary_device.hpp` | `enqueue_wall_ghosts`, one launch per walled axis in x, y, z order | `KernelLaunch<>` over the face samples of the axis | `lpfx::device::for_each_grid_index` | no |
| `constrained_rows_kernel<Axis, ...>` | same | `enqueue_constrained_rows`, one launch per axis with constrained faces in x, y, z order | `KernelLaunch<>` over the face rows of the axis | `lpfx::device::for_each_grid_index` | no |
| `field_vector_linear_combination_kernel` | `ecsim/fields/field_vector_device.hpp` | `enqueue_field_vector_linear_combination` | `KernelLaunch<>` over the samples | `lpfx::device::for_each_grid_index`; left fold of the terms, one narrowing | yes |
| `field_vector_copy_kernel` | same | `enqueue_field_vector_copy` | `KernelLaunch<>` over the samples | `lpfx::device::for_each_grid_index` | yes |
| `node_cell_curl_kernel<Region, ...>` | `ecsim/fields/node_cell_curl_device.hpp` | `enqueue_node_cell_curl` | `KernelLaunch<>` over `stencil_region_count<Region>`, `EmptyWork::skip` | `lpfx::device::for_each_grid_index` | no |
| `node_smoothing_kernel<Operator, ...>` | `ecsim/fields/node_smoothing_device.hpp` | `enqueue_node_smoothing` (`Face7` or `Binomial27`) | `KernelLaunch<>` over the interior nodes | `lpfx::device::for_each_grid_index` | yes |

## ecsim/gmres

All are private to `Gmres` in `ecsim/gmres/gmres_device.hpp` and launched with
`lpfx::device::launch_or_throw`; the reduction grid has `gmres_reduction_blocks`
(`ecsim/gmres/gmres.hpp`) blocks.

| Kernel | Launched by | Shape and work | Iteration | Flag |
| --- | --- | --- | --- | --- |
| `gmres_field_vector_transfer_kernel<Direction, ...>` | `pack`, `unpack` | `GmresLaunch` over the interior nodes | `lpfx::device::for_each_grid_index` | no |
| `gmres_dot_batch_kernel` | `enqueue_projection_dots` | Two-dimensional grid: `gmres_reduction_blocks` vector segments by tiles of `gmres_dot_tile` basis vectors | `lpfx::device::for_each_grid_index` per segment; warp shuffles and a shared exchange, one partial per block and column | no |
| `gmres_residual_squared_norm_kernel<RhsNorm, ...>` | `enqueue_residual_squared_norm` | `LaunchGrid{gmres_reduction_blocks, threads}` | `lpfx::device::for_each_grid_index`; one partial per block and sum | no |
| `gmres_normalize_kernel<..., Tail, Store>` | `normalize` | `LaunchGrid{gmres_reduction_blocks, threads}`, so few threads form the FP64 scale | `lpfx::device::for_each_grid_index`; the tail runs on block 0, thread 0 | no |
| `gmres_orthogonalize_squared_norm_kernel` | `host_blocking_solve` | `LaunchGrid{gmres_reduction_blocks, threads}` | `lpfx::device::for_each_grid_index`; one partial per block | no |
| `gmres_back_substitute_kernel` | `host_blocking_solve` | `LaunchGrid{1, 1}` | Single thread | no |
| `gmres_update_solution_kernel` | `host_blocking_solve` | `GmresLaunch` over the vector | `lpfx::device::for_each_grid_index` | no |

## ecsim/mass_matrix

| Kernel | Header | Launched by | Shape and work | Iteration | Flag |
| --- | --- | --- | --- | --- | --- |
| `mass_matrix_apply_kernel<Precision, Compute, Region, ...>` | `ecsim/mass_matrix/mass_matrix_apply_device.hpp` | `enqueue_mass_matrix_apply` | `KernelLaunch<mass_matrix_block_threads>` over `stencil_region_count<Region>`, `EmptyWork::skip` | `lpfx::device::for_each_grid_index` | yes |
| `unsorted_mass_matrix_kernel` | `ecsim/mass_matrix/mass_matrix_scalar_device.hpp` | `enqueue_unsorted_mass_matrix` | `KernelLaunch<particle_block_threads>` over the particle range, `EmptyWork::skip` | `lpfx::device::for_each_grid_index`, one particle per index | yes |
| `sorted_mass_matrix_scalar_kernel` | same | `detail::launch_sorted_mass_matrix`, from `enqueue_sorted_mass_matrix` and `enqueue_resident_mass_matrix` (`MassDepositionKernel::scalar`) | `SortedCellLaunch` over the cell bins | One subgroup per cell bin, stride over bins (`for_each_sorted_cell_broadcast`: lane 0 opens the bin and broadcasts it); shared magnetic cache | yes |
| `sorted_mass_matrix_mma_kernel` | `ecsim/mass_matrix/mass_matrix_mma_device.hpp` | `detail::launch_sorted_mass_matrix` (`matrix_core`, CUDA) | `SortedCellLaunch` over the cell bins | One warp per cell bin as in the scalar kernel, 64-particle register chunks, FP64 accumulators narrowed at the pair flush | yes |
| `sorted_mass_matrix_mfma4_kernel` | `ecsim/mass_matrix/mass_matrix_mfma_device.hpp` | `detail::launch_sorted_mass_matrix` (`matrix_core`, HIP) | `MassMfma4Launch` over the cell bins | One wave64 per cell bin as in the scalar kernel, four per block, 64-particle chunks of MFMA4 instructions | yes |

`detail::launch_sorted_mass_matrix` selects the kernel at compile time from
`MassDepositionKernel` and the backend; there is no run-time kernel choice.
The matrix-core instructions of the MMA and MFMA4 kernels, with the MFMA4
instruction name and its modifier arguments, are described in the
[numerical reference](numerical_reference.md).

## ecsim/maxwell and ecsim/particles

| Kernel | Header | Launched by | Shape and work | Iteration | Flag |
| --- | --- | --- | --- | --- | --- |
| `faraday_update_kernel` | `ecsim/maxwell/field_update_device.hpp` | `enqueue_next_fields`, from `MaxwellOperator` | `KernelLaunch<>` over all cells | `lpfx::device::for_each_grid_index`; node-to-cell curl in registers, one narrowing | yes |
| `unsorted_particle_mover_kernel` | `ecsim/particles/particle_mover_device.hpp` | `detail::launch_particle_mover`, from `enqueue_migrating_particle_mover` and `enqueue_local_particle_mover` | `KernelLaunch<>` over the particles after the sorted prefix (all of them without a sort receipt) | `lpfx::device::for_each_grid_index`, one particle per index | yes |
| `sorted_particle_mover_kernel` | same | `detail::launch_particle_mover`, when a sort receipt is passed | `SortedCellLaunch` over `interior_cell_count`; the unsorted kernel then moves the tail | One subgroup per interior cell bin, stride over bins; halo bins get no subgroup | yes |

## initialization

| Kernel | Header | Launched by | Shape and work | Iteration | Flag |
| --- | --- | --- | --- | --- | --- |
| `center_particle_velocity_kernel` (two instantiations per case profile: drift removal, then restoring correction) | `initialization/particles/particle_centering_device.hpp` | `VelocityCentering::enqueue_step` | `LaunchGrid{KernelLaunch<>::blocks_at_least_one(particles), threads}`: at least one block, so every rank takes part | `lpfx::device::for_each_grid_index` | yes |
| `initial_particle_load_kernel` (two instantiations per case profile, subcell centres or random offsets, selected by `QuietStart`; every local subcell, or the uploaded rows of a bounded support) | `initialization/particles/particle_loading_device.hpp` | `enqueue_initial_particle_load` | `KernelLaunch<>` over the load range, `EmptyWork::skip` | `lpfx::device::for_each_grid_index` | yes |

## Keeping this list current

A kernel gets its row here in the same change that adds it, with its header,
launching function, shape and iteration model. The list can be checked against
the tree with `git grep -n __global__ -- include/tpic`; `include/` also holds
the LPFx kernels.
