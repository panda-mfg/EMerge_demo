"""Benchmark demo23's CPU Pardiso and NVIDIA cuDSS direct solvers.

Run each backend in a fresh process so solver memory and state cannot carry
between measurements::

    python examples/benchmark_demo23_high_pass_birdcage.py \
        --solver pardiso --cpu-threads 12 --output /tmp/birdcage-pardiso.json
    python examples/benchmark_demo23_high_pass_birdcage.py \
        --solver cudss --output /tmp/birdcage-cudss.json

The benchmark uses demo23's complete 118--138 MHz, 11-point sweep, tuned
24.1 pF capacitors, second-order mixed basis, and two-port excitation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import threading
import time
from pathlib import Path


MIB = 1024**2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("pardiso", "cudss"), required=True)
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cpu_model_name():
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        for line in cpuinfo:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def complex_records(values):
    return [{"real": float(value.real), "imag": float(value.imag)} for value in values]


def monitor_gpu_memory(stop_event, samples):
    import cupy as cp

    while not stop_event.is_set():
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        samples.append(total_bytes - free_bytes)
        stop_event.wait(0.05)


def main():
    args = parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")

    # Pardiso reads OMP_NUM_THREADS while its interface initializes. Set both
    # common MKL controls before importing EMerge.
    cpu_threads = args.cpu_threads if args.solver == "pardiso" else 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ.setdefault("NUMBA_NUM_THREADS", "4")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import numpy as np

    import demo23_high_pass_birdcage as birdcage

    solver = {
        "pardiso": birdcage.em.EMSolver.PARDISO,
        "cudss": birdcage.em.EMSolver.CUDSS,
    }[args.solver]

    process_start = time.perf_counter()
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)

    build_start = time.perf_counter()
    params = birdcage.BirdcageParameters()
    model, _, _ = birdcage.build_model(params)
    model.set_solver(solver)
    build_seconds = time.perf_counter() - build_start

    gpu_stop = threading.Event()
    gpu_samples = []
    gpu_thread = None
    if args.solver == "cudss":
        gpu_thread = threading.Thread(
            target=monitor_gpu_memory,
            args=(gpu_stop, gpu_samples),
            daemon=True,
        )
        gpu_thread.start()

    sweep_start = time.perf_counter()
    data = model.mw.run_sweep()
    sweep_seconds = time.perf_counter() - sweep_start

    if gpu_thread is not None:
        gpu_stop.set()
        gpu_thread.join()

    grid = data.scalar.grid
    frequencies = np.asarray(grid.freq)
    s11 = np.asarray(grid.S(1, 1))
    s21 = np.asarray(grid.S(2, 1))
    s22 = np.asarray(grid.S(2, 2))

    solve_reports = []
    for variables, entry in data.sim.iterate():
        for report in entry["report"]:
            solve_reports.append(
                {
                    "frequency_hz": float(variables["freq"]),
                    "solver": report.solver,
                    "seconds": float(report.simtime),
                    "ndof": int(report.ndof_solve),
                    "nnz": int(report.nnz_solve),
                }
            )

    cpu_stop = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "solver": args.solver,
        "cpu_threads": cpu_threads,
        "numba_threads": int(os.environ["NUMBA_NUM_THREADS"]),
        "python_version": platform.python_version(),
        "emerge_version": package_version("emerge"),
        "numpy_version": np.__version__,
        "cpu": cpu_model_name(),
        "gpu": None,
        "cupy_version": None,
        "nvmath_version": None,
        "cudss_package_version": None,
        "cuda_runtime_version": None,
        "cuda_driver_api_version": None,
        "gpu_total_memory_mib": None,
        "mesh_tetrahedra": int(model.mesh.n_tets),
        "frequency_hz": frequencies.tolist(),
        "capacitance_pf": params.capacitance / birdcage.pF,
        "build_seconds": build_seconds,
        "sweep_seconds": sweep_seconds,
        "process_seconds": time.perf_counter() - process_start,
        "solver_seconds_total": sum(item["seconds"] for item in solve_reports),
        "solver_seconds_by_frequency": solve_reports,
        "peak_host_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "process_cpu_seconds": (cpu_stop.ru_utime - cpu_start.ru_utime)
        + (cpu_stop.ru_stime - cpu_start.ru_stime),
        "gpu_baseline_used_mib": None,
        "gpu_peak_used_mib": None,
        "gpu_peak_delta_mib": None,
        "s11": complex_records(s11),
        "s21": complex_records(s21),
        "s22": complex_records(s22),
    }

    if args.solver == "cudss":
        import cupy as cp

        device = cp.cuda.runtime.getDeviceProperties(0)
        result["gpu"] = device["name"].decode()
        result["cupy_version"] = cp.__version__
        result["nvmath_version"] = package_version("nvmath-python")
        result["cudss_package_version"] = package_version("nvidia-cudss-cu12")
        result["cuda_runtime_version"] = cp.cuda.runtime.runtimeGetVersion()
        result["cuda_driver_api_version"] = cp.cuda.runtime.driverGetVersion()
        result["gpu_total_memory_mib"] = device["totalGlobalMem"] / MIB
        if gpu_samples:
            result["gpu_baseline_used_mib"] = gpu_samples[0] / MIB
            result["gpu_peak_used_mib"] = max(gpu_samples) / MIB
            result["gpu_peak_delta_mib"] = (max(gpu_samples) - gpu_samples[0]) / MIB

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")

    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"s11", "s21", "s22", "solver_seconds_by_frequency"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
