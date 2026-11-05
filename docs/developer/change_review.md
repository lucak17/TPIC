# Change review

A change to TPIC is qualified by its class. The class fixes the checks it must
pass so that the [engineering standards](standards.md), R1 above all, hold. The
[kernel inventory](kernels.md) lists the kernels whose bodies and launches the
classes K and L cover; [testing](testing.md) describes the test suites and
[performance](performance.md) the timing and profiling tools.

## Checks

The checks are: host build and host tests; device build and device tests;
kernel and launch source identity (token streams with comments stripped);
identity of the generated device code per kernel instantiation (modulo symbol
names); identity of the host symbols; the static scheduling sequence (streams,
events, waits, records, fences, communication calls); the runtime call sequence
of a traced run (launches with grid, block, shared memory and stream;
allocations; MPI and NCCL/RCCL calls); run equivalence of fields, moments,
scalars and particles within the measured envelope; timing per stage row
against the measured null band; console text and output file bytes.

## Change classes

Every change declares the class of each of its commits.

| Class | Definition | Checks |
| --- | --- | --- |
| A, host only | Touches only host-compiled sources, tests, tools, documents and headers that no device translation unit includes. | Host and device builds and tests; console text when the code prints; output bytes when it writes files. |
| B, move, rename or delete | Moves text between files, renames identifiers, deletes declarations that no production code uses, or adds `static_assert` declarations. No statement of device-compiled code changes. | Class A checks, source identity, device code identity, host symbol identity, scheduling sequence; once per change the traced call sequence, a reduced run comparison and one timing case. |
| C, host orchestration | Changes statements in device-compiled host code outside every function that contains a launch and outside every expression that produces a launch argument. | Class B checks, the traced call sequence and console text on every commit; full run equivalence and timing at the end. |
| D, launch-adjacent | Edits a function that contains a launch, code that produces a launch argument or the template arguments of a launch; renames a kernel or a kernel argument type; or rewrites a launch through `lpfx::device::launch`. Kernel bodies and launch configurations do not change. | Class C checks; device code identity on every target architecture; differences in source and trace limited to the listed lines; `tests/ecsim/device_submission_rejection.cpp`. |
| E, instantiation | Changes which templates are instantiated or the type of an object that device-compiled code holds. | Class D checks; the kernel instantiation set and the number of `Simulation` instantiations unchanged; build time and size recorded. |
| K, kernel body | Edits the body, template head or signature of a listed kernel or of a device function it calls. Launch configurations do not change. | Class D checks; device code identical where identical arithmetic is claimed, otherwise differing only in the listed instantiations; per-kernel performance no worse for every changed instantiation; numerical equivalence of the kernel and long energy-conservation runs; full timing. |
| L, launch configuration | Changes grid, block, shared-memory size, `__launch_bounds__`, the number of launches of a kernel family or the size of a buffer only that family uses. | Class K checks; the traced geometry and launch count differ only as listed; timing on every case. Accepted only when every listed kernel and every stage row is faster or equal. |
| G, graph change | Removes a record, an event nothing waits on, a host fence nothing waits on, a startup collective or a startup agreement, or changes the payload type of a startup collective. Nothing is added. | Class C checks; scheduling sequence and trace differ only as listed; full timing, and startup time for startup changes. Accepted only when no case is slower. |

No class adds a record, a wait, a synchronisation or a communication call, and
only class G removes one or moves it. Deleting a wrapper that only forwards to
such a call, so that the caller names the same callee with the same arguments
at the same place, is allowed in classes C and D. A change that needs more is a
design change, measured on its own. HIP text falls into the same classes as
CUDA text; a class K or L edit of HIP text also shows HIP resource usage (registers,
LDS, scratch, occupancy) no worse, and is measured on AMD hardware before
release.

## Removing a check inside a kernel

A check inside a kernel (a finiteness test, a range test, a validation flag
write) is part of how TPIC detects failure. A class K change may remove one
only when a named test shows one of these conditions, and it states which
inputs the check caught.

- **Implied.** Another check that stays catches the same inputs at the same or
  an earlier point, before any result of the step is accepted. Example: a
  non-finite magnetic field or charge propagates to the current product, whose
  check stays.
- **Impossible.** The inputs cannot occur by construction, stated as a
  `static_assert` or an `if constexpr` condition. Example: a finite value times
  a weight in [0, 1] cannot overflow when the storage type equals the compute
  type.
- **Fatal elsewhere.** A later check is fatal on every rank before any result of
  the step is used.

In every case a fault-injection test (non-finite field, charge or velocity;
fraction outside [0, 1); corrupted bin offsets) passes before and after and
still raises the flag. A removal that changes which record is retained or moved
on failure is a behaviour change and is documented for users.
