from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np


def _ensure_gpu_available():
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError("No GPU found or CUDA is not configured.") from exc


def _build_kernel_source(params: dict, kernel_path: Path) -> str:
    defines = f"""
#include <cuComplex.h>

#define M_PI 3.14159265358979323846
#define MILLER_MARGIN {params["miller_margin"]}
#define MILLER_MIN_MARGIN {params["miller_min_margin"]}
#define MILLER_MAX_START {params["miller_max_start"]}
#define TEMP_ARRAY_SIZE {params["temp_array_size"]}
#define BESSEL_ARRAY_SIZE {params["bessel_array_size"]}
#define SPECTRAL_ARRAY_SIZE {params["spectral_array_size"]}
#define SERIES_MAX_ITER {params["series_max_iter"]}
"""
    kernel_body = kernel_path.read_text(encoding="utf-8")
    return defines + kernel_body


# Cache for compiled kernels
_kernel_cache = {}


def _load_kernel(params: dict, kernel_path: Path, kernel_name: str):
    """Load and cache a CUDA kernel."""
    include_dir = str(kernel_path.parent)
    cache_key = (tuple(sorted(params.items())), str(kernel_path), kernel_name, include_dir)
    if cache_key not in _kernel_cache:
        source = _build_kernel_source(params, kernel_path)
        _kernel_cache[cache_key] = cp.RawKernel(source, kernel_name, options=(f"-I{include_dir}",))
    return _kernel_cache[cache_key]


def compute_q_and_grad(
    eps_re: float,
    eps_im: float,
    radius: float,
    wavelengths: np.ndarray,
    n_max: int,
    kernel_params: dict,
    kernel_path: Path | None = None,
) -> dict:

    _ensure_gpu_available()
    
    if kernel_path is None:
        kernel_path = Path(__file__).with_name("mie_kernel_grad_generated.cu")
    
    kernel = _load_kernel(kernel_params, kernel_path, "mie_cylinder_grad_kernel")

    wavelengths_gpu = cp.asarray(wavelengths, dtype=cp.float64)
    delta_wav = float(wavelengths[-1] - wavelengths[0])

    out_q_tm = cp.zeros((1,), dtype=cp.float64)
    out_q_te = cp.zeros((1,), dtype=cp.float64)
    out_grad_tm = cp.zeros((3,), dtype=cp.float64)
    out_grad_te = cp.zeros((3,), dtype=cp.float64)

    kernel(
        (1,),
        (1,),
        (
            float(radius),
            float(eps_re),
            float(eps_im),
            wavelengths_gpu,
            len(wavelengths),
            int(n_max),
            float(delta_wav),
            out_q_tm,
            out_q_te,
            out_grad_tm,
            out_grad_te,
        ),
    )

    cp.cuda.Stream.null.synchronize()

    return {
        "q_tm": float(out_q_tm.get()[0]),
        "q_te": float(out_q_te.get()[0]),
        "grad_tm": out_grad_tm.get().astype(float),
        "grad_te": out_grad_te.get().astype(float),
    }


def run_batched_gradient_descent(
    radii: np.ndarray,
    wavelengths: np.ndarray,
    n_max: int,
    kernel_params: dict,
    eps_re_min: float = -5.0,
    eps_re_max: float = 10.0,
    eps_im_min: float = -10.0,
    eps_im_max: float = -0.5,
    learning_rate: float = 0.1,
    momentum: float = 0.9,
    lr_decay: float = 0.98,
    max_iters: int = 50,
    tolerance: float = 1e-8,
    n_restarts: int = 3,
    use_tm: bool = True,
    kernel_path: Path | None = None,
) -> dict:
    
    _ensure_gpu_available()
    
    if kernel_path is None:
        kernel_path = Path(__file__).with_name("mie_kernel_grad_batch_generated.cu")
    
    kernel = _load_kernel(kernel_params, kernel_path, "mie_gradient_descent_kernel")
    
    n_radii = len(radii)
    total_threads = n_radii * n_restarts
    
    radii_gpu = cp.asarray(radii, dtype=cp.float64)
    wavelengths_gpu = cp.asarray(wavelengths, dtype=cp.float64)
    delta_wav = float(wavelengths[-1] - wavelengths[0])
    
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**32, size=total_threads, dtype=np.uint32)
    seeds_gpu = cp.asarray(seeds)
    
    out_q = cp.zeros(total_threads, dtype=cp.float64)
    out_eps_re = cp.zeros(total_threads, dtype=cp.float64)
    out_eps_im = cp.zeros(total_threads, dtype=cp.float64)
    
    block_size = 256
    grid_size = (total_threads + block_size - 1) // block_size
    
    kernel(
        (grid_size,),
        (block_size,),
        (
            radii_gpu,
            np.int32(n_radii),
            np.int32(n_restarts),
            wavelengths_gpu,
            np.int32(len(wavelengths)),
            np.int32(n_max),
            np.float64(delta_wav),
            np.float64(eps_re_min),
            np.float64(eps_re_max),
            np.float64(eps_im_min),
            np.float64(eps_im_max),
            np.float64(learning_rate),
            np.float64(momentum),
            np.float64(lr_decay),
            np.int32(max_iters),
            np.float64(tolerance),
            np.bool_(use_tm),
            seeds_gpu,
            out_q,
            out_eps_re,
            out_eps_im,
        ),
    )
    
    cp.cuda.Stream.null.synchronize()
    
    q_all = out_q.get().reshape(n_radii, n_restarts)
    eps_re_all = out_eps_re.get().reshape(n_radii, n_restarts)
    eps_im_all = out_eps_im.get().reshape(n_radii, n_restarts)
    
    best_restart_idx = np.argmax(q_all, axis=1)
    best_q = q_all[np.arange(n_radii), best_restart_idx]
    best_eps_re = eps_re_all[np.arange(n_radii), best_restart_idx]
    best_eps_im = eps_im_all[np.arange(n_radii), best_restart_idx]
    
    return {
        "radii": radii,
        "metric": best_q,
        "eps_re": best_eps_re,
        "eps_im": best_eps_im,
    }
