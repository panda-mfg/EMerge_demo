# EMerge demo examples

This repository is a focused collection of runnable examples for
[EMerge](https://github.com/FennisRobert/EMerge), a Python finite-element
method (FEM) simulator for frequency-domain electromagnetic and heat-transfer
problems.

The repository contains example scripts and the geometry asset used by the
STEP-import example. It does **not** contain the EMerge package itself,
precomputed simulation results, or the full upstream source tree.

## Requirements

- Python 3.10 through 3.13
- EMerge 2.8.4 (the version checked by the electromagnetic examples)
- A supported direct solver for the selected example
- A graphical environment for examples that open interactive plots or 3D views

Create a virtual environment and install the matching EMerge release:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "emerge==2.8.4"
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
See the [upstream installation notes](https://github.com/FennisRobert/EMerge#how-to-install)
for platform-specific solver guidance.

## Running an example

Run scripts from the repository root. For example:

```bash
python examples/demo0_parallel_plate.py
```

The STEP-import example automatically loads `examples/DielectricRod.step`
relative to its own location:

```bash
python examples/demo17_step_import.py
```

Many examples generate a mesh, solve one or more FEM systems, and open
interactive Matplotlib or PyVista windows. Runtime and memory use vary widely;
the optimization and MRI birdcage examples are among the more demanding cases.

## Example catalog

| Example | Topic |
| --- | --- |
| `demo0_parallel_plate.py` | Introductory parallel-plate model |
| `demo1_stepped_imp_filter.py` | Stepped-impedance filter |
| `demo2_combline_filter.py` | Combline filter |
| `demo3_coupled_line_filter.py` | Coupled-line filter |
| `demo4_patch_antenna.py` | Microstrip patch antenna |
| `demo5_revolve.py` | Geometry construction by revolving a profile |
| `demo6_striplines_with_vias.py` | Striplines and vias |
| `demo7_periodic_cells.py` | Periodic structures and unit cells |
| `demo8_waveguide_bpf_synthesis.py` | Waveguide band-pass filter synthesis |
| `demo9_dielectric_resonator.py` | Dielectric resonator model |
| `demo10_sgh.py` | Standard-gain horn antenna |
| `demo11_lumped_element_filter.py` | Lumped-element filter |
| `demo12_mode_alignment.py` | Port-mode alignment |
| `demo13_helix_antenna.py` | Helical antenna |
| `demo14_boundary_selection.py` | Geometry face and boundary selection |
| `demo15_strip_slotline_transition.py` | Stripline-to-slotline transition |
| `demo16_differential_common_mode.py` | Differential- and common-mode excitation |
| `demo17_step_import.py` | STEP geometry import and dielectric-rod antenna |
| `demo18_plotting_and_visualization.py` | Plotting and visualization APIs |
| `demo19_vivaldi_antenna.py` | Vivaldi antenna |
| `demo20_optimization.py` | Parameter optimization workflow |
| `demo21_inverted_F_antenna.py` | Inverted-F antenna |
| `demo22_RCS.py` | Radar cross-section simulation |
| `demo23_high_pass_birdcage.py` | 128 MHz high-pass MRI birdcage coil |
| `benchmark_demo23_high_pass_birdcage.py` | CPU PARDISO and NVIDIA cuDSS birdcage benchmark |

The `examples/heatconduction/` directory contains five additional demos:

- Basic steady-state heat simulation
- Chip heating
- RF heating
- Black-body radiation
- Transient heat conduction

## MRI birdcage and benchmark

Inspect the birdcage mesh without running the complete frequency sweep:

```bash
python examples/demo23_high_pass_birdcage.py --mesh-only --view
```

Run its CPU benchmark and write the measurements to JSON:

```bash
python examples/benchmark_demo23_high_pass_birdcage.py \
  --solver pardiso \
  --cpu-threads 12 \
  --output birdcage-pardiso.json
```

The benchmark wrapper uses Linux process and CPU information interfaces. The
cuDSS backend also requires a compatible NVIDIA GPU, CUDA environment, and the
optional EMerge dependencies:

```bash
python -m pip install "emerge[cudss]==2.8.4"
python examples/benchmark_demo23_high_pass_birdcage.py \
  --solver cudss \
  --output birdcage-cudss.json
```

## Repository layout

```text
.
├── README.md
└── examples
    ├── DielectricRod.step
    ├── demo*.py
    ├── benchmark_demo23_high_pass_birdcage.py
    └── heatconduction
        └── demo*.py
```

## Upstream project

For the EMerge source, full installation guidance, documentation, issue
tracking, and licensing terms, visit the
[upstream EMerge repository](https://github.com/FennisRobert/EMerge).
