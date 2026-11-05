# TPIC

TilePIC (TPIC) is an implicit energy-conserving particle-in-cell code for CUDA and HIP GPUs.
It advances particles and electromagnetic fields with the energy-conserving semi-implicit
method (ECSIM) and distributes the domain over MPI ranks, each driving one GPU.

The LPFx header library supplies the typed storage, execution and
communication primitives. It is bundled in `include/lpfx` and `cmake/lpfx`;
`cmake/dependencies.json` records its revision and the hash of every file
copied unchanged, and `python3 tools/dependency_inventory.py --check` verifies
the copy.

Requirements: CMake 3.24, a C++20 compiler and MPI for every build; CUDA with
cuRAND or HIP with hipRAND for a device build. Versions and optional packages
are in [build and run](docs/user/build_and_run.md).

## Quick start

Host build: input checker and portable tests (no CPU time step):

```sh
cmake --preset host
cmake --build --preset host -j 4
ctest --preset host
build/host/tpic --check-input input/uniformStartup.inp
```

Device build: the architecture is required, `120` for an RTX 5080, `80` for an
A100, `90` for a GH200, or `hip` with `gfx90a` for an MI250X:

```sh
cmake --preset cuda -DLPFX_DEVICE_ARCHITECTURES=120
cmake --build --preset cuda -j 2
mpiexec -n 1 build/cuda/tpic --run input/uniformStartup.inp
```

```sh
cmake --preset hip -DLPFX_DEVICE_ARCHITECTURES=gfx90a
cmake --build --preset hip -j 2
```

The first run advances a small uniform periodic plasma by three cycles on one
rank, prints the console report and writes no files; it only creates the
empty output folder `data`. The `cuda` and `hip` presets pass device buffers
to MPI and so need a GPU-aware MPI; NCCL/RCCL transport and the other options
are in [build and run](docs/user/build_and_run.md).
`tpic --help` lists the commands and options.

Without presets, the same CUDA build with the options written out:

```sh
cmake -S . -B build/cuda -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTPIC_BUILD_TESTS=ON \
  -DLPFX_DEVICE_BACKEND=CUDA -DLPFX_DEVICE_ARCHITECTURES=120 \
  -DLPFX_ENABLE_MPI=ON -DLPFX_ENABLE_DEVICE_AWARE_MPI=ON \
  -DTPIC_COMMUNICATION_BACKEND=MPI \
  -DTPIC_ENABLE_ADIOS2=ON -DTPIC_ENABLE_HDF5=ON
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure
```

HIP uses `-DLPFX_DEVICE_BACKEND=HIP -DLPFX_DEVICE_ARCHITECTURES=gfx90a` (the
target of the GPU); the last line of options (ADIOS2, HDF5) is
optional. Every option is in [build and run](docs/user/build_and_run.md).

Every deck directly in `input/` is a complete template: it lists every supported
general option with its choices, optional features off. Copy the deck closest to the
planned run and edit it; [inputs](docs/user/inputs.md) explains the syntax and
[cases](docs/user/cases.md) the shipped cases.

## Documentation

The [documentation index](docs/README.md) lists the user and developer guides;
[scaling](performance/scaling/README.md) covers cluster campaigns.

## Licence

TPIC is distributed under the GNU General Public License, version 3
([LICENSE](LICENSE)).

## How to cite

Cite TPIC:

- L. Pennati, I. Peng and S. Markidis, "TPIC: Energy-Conserving Space Plasma
  Simulations on GPU Supercomputers", arXiv:2611.09624 (2026).

```bibtex
@misc{pennati2026tpic,
  title={TPIC: Energy-Conserving Space Plasma Simulations on GPU Supercomputers},
  author={Luca Pennati and Ivy Peng and Stefano Markidis},
  year={2026},
  eprint={2611.09624},
  archivePrefix={arXiv},
  primaryClass={cs.CE},
  url={https://arxiv.org/abs/2611.09624},
}
```

Cite the ECSIM method:

- G. Lapenta, "Exactly energy conserving semi-implicit particle in cell
  formulation", J. Comput. Phys. 334, 349 (2017).

For the matrix-core (tensor-core) mass-matrix assembly (`--mass-matrix matrix-core`):

- L. Pennati and S. Markidis, "Mass matrix assembly on tensor cores for
  implicit particle-in-cell methods", J. Comput. Phys. 569, 115442 (2027),
  doi:10.1016/j.jcp.2026.115442.
