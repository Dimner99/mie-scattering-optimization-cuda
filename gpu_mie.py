from pathlib import Path
from contextlib import contextmanager

import numpy as np
import cupy as cp
from tqdm import tqdm

try:
    from cupy.cuda import nvtx as _nvtx
except Exception: 
    _nvtx = None


@contextmanager
def nvtx_range(message):
    if _nvtx is None:
        yield
        return
    _nvtx.RangePush(message)
    try:
        yield
    finally:
        _nvtx.RangePop()


def build_cuda_source(
    miller_margin,
    miller_min_margin,
    miller_max_start,
    temp_array_size,
    bessel_array_size,
    spectral_array_size,
    series_max_iter,
):
    defines = f"""
#include <cuComplex.h>

#define M_PI 3.14159265358979323846
#define MILLER_MARGIN {miller_margin}
#define MILLER_MIN_MARGIN {miller_min_margin}
#define MILLER_MAX_START {miller_max_start}
#define TEMP_ARRAY_SIZE {temp_array_size}
#define BESSEL_ARRAY_SIZE {bessel_array_size}
#define SPECTRAL_ARRAY_SIZE {spectral_array_size}
#define SERIES_MAX_ITER {series_max_iter}
"""

    kernel_path = Path(__file__).with_name("mie_kernel.cu")
    kernel_body = kernel_path.read_text()
    return defines + kernel_body


def load_kernel(params):
    source = build_cuda_source(
        miller_margin=params["miller_margin"],
        miller_min_margin=params["miller_min_margin"],
        miller_max_start=params["miller_max_start"],
        temp_array_size=params["temp_array_size"],
        bessel_array_size=params["bessel_array_size"],
        spectral_array_size=params["spectral_array_size"],
        series_max_iter=params["series_max_iter"],
    )
    include_dir = str(Path(__file__).parent)
    return cp.RawKernel(source, "mie_cylinder_kernel", options=(f"-I{include_dir}",))


def ensure_gpu_available():
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError("No GPU found or CUDA is not configured.") from exc


def run_gpu_sweep(
    wavelengths,
    radii_tm,
    radii_te,
    eps_re_tm,
    eps_im_tm,
    eps_re_te,
    eps_im_te,
    n_max,
    kernel_params,
    compute_tm=True,
    compute_te=True,
    show_progress=True,
    nvtx=False,
):
    ensure_gpu_available()
    kernel = load_kernel(kernel_params)
    block_size = (16, 16)

    wavelengths_gpu = cp.asarray(wavelengths)
    delta_wav = float(wavelengths[-1] - wavelengths[0])

    results = {}

    if compute_tm:
        eps_re_tm_gpu = cp.asarray(eps_re_tm)
        eps_im_tm_gpu = cp.asarray(eps_im_tm)

        grid_size_tm = (
            (len(eps_re_tm) + block_size[0] - 1) // block_size[0],
            (len(eps_im_tm) + block_size[1] - 1) // block_size[1],
        )

        best_tm_abs = cp.full((len(eps_re_tm), len(eps_im_tm)), -cp.inf, dtype=cp.float64)
        best_tm_a = cp.zeros((len(eps_re_tm), len(eps_im_tm)), dtype=cp.float64)
        out_tm_map_gpu = cp.zeros((len(eps_re_tm), len(eps_im_tm)), dtype=cp.float64)
        out_te_map_gpu_dummy = cp.zeros((len(eps_re_tm), len(eps_im_tm)), dtype=cp.float64)

        radii_iter_tm = (
            tqdm(radii_tm, desc="Sweeping Radii (TM) - GPU") if show_progress else radii_tm
        )
        with nvtx_range("TM sweep") if nvtx else nvtx_range(""):
            for a_val in radii_iter_tm:
                with nvtx_range("TM kernel") if nvtx else nvtx_range(""):
                    kernel(
                        grid_size_tm,
                        block_size,
                        (
                            float(a_val),
                            eps_re_tm_gpu,
                            eps_im_tm_gpu,
                            wavelengths_gpu,
                            len(eps_re_tm),
                            len(eps_im_tm),
                            len(wavelengths),
                            n_max,
                            delta_wav,
                            out_tm_map_gpu,
                            out_te_map_gpu_dummy,
                        ),
                    )
                    # Removed per-kernel sync - only sync when we need CPU access
                mask_tm = out_tm_map_gpu > best_tm_abs
                best_tm_abs[mask_tm] = out_tm_map_gpu[mask_tm]
                best_tm_a[mask_tm] = a_val
        
        # Synchronize once before transferring data back to CPU
        cp.cuda.Stream.null.synchronize()

        results["tm"] = {
            "best_abs": best_tm_abs.get(),
            "best_a": best_tm_a.get(),
            "re_mesh": np.meshgrid(eps_re_tm, eps_im_tm, indexing="ij")[0],
            "im_mesh": np.meshgrid(eps_re_tm, eps_im_tm, indexing="ij")[1],
        }
    else:
        results["tm"] = None

    if compute_te:
        eps_re_te_gpu = cp.asarray(eps_re_te)
        eps_im_te_gpu = cp.asarray(eps_im_te)

        grid_size_te = (
            (len(eps_re_te) + block_size[0] - 1) // block_size[0],
            (len(eps_im_te) + block_size[1] - 1) // block_size[1],
        )

        best_te_abs = cp.full((len(eps_re_te), len(eps_im_te)), -cp.inf, dtype=cp.float64)
        best_te_a = cp.zeros((len(eps_re_te), len(eps_im_te)), dtype=cp.float64)
        out_tm_map_gpu_dummy = cp.zeros((len(eps_re_te), len(eps_im_te)), dtype=cp.float64)
        out_te_map_gpu = cp.zeros((len(eps_re_te), len(eps_im_te)), dtype=cp.float64)

        radii_iter_te = (
            tqdm(radii_te, desc="Sweeping Radii (TE) - GPU") if show_progress else radii_te
        )
        with nvtx_range("TE sweep") if nvtx else nvtx_range(""):
            for a_val in radii_iter_te:
                with nvtx_range("TE kernel") if nvtx else nvtx_range(""):
                    kernel(
                        grid_size_te,
                        block_size,
                        (
                            float(a_val),
                            eps_re_te_gpu,
                            eps_im_te_gpu,
                            wavelengths_gpu,
                            len(eps_re_te),
                            len(eps_im_te),
                            len(wavelengths),
                            n_max,
                            delta_wav,
                            out_tm_map_gpu_dummy,
                            out_te_map_gpu,
                        ),
                    )
                mask_te = out_te_map_gpu > best_te_abs
                best_te_abs[mask_te] = out_te_map_gpu[mask_te]
                best_te_a[mask_te] = a_val
        
        cp.cuda.Stream.null.synchronize()

        results["te"] = {
            "best_abs": best_te_abs.get(),
            "best_a": best_te_a.get(),
            "re_mesh": np.meshgrid(eps_re_te, eps_im_te, indexing="ij")[0],
            "im_mesh": np.meshgrid(eps_re_te, eps_im_te, indexing="ij")[1],
        }
    else:
        results["te"] = None

    return results


def run_gpu_1d_optimization(
    wavelengths,
    radii_tm,
    radii_te,
    eps_re_tm,
    eps_im_tm,
    eps_re_te,
    eps_im_te,
    n_max,
    kernel_params,
    compute_tm=True,
    compute_te=True,
    show_progress=True,
    nvtx=False,
):
    ensure_gpu_available()
    kernel = load_kernel(kernel_params)
    block_size = (16, 16)

    wavelengths_gpu = cp.asarray(wavelengths)
    delta_wav = float(wavelengths[-1] - wavelengths[0])

    results = {}

    if compute_tm:
        eps_re_tm_gpu = cp.asarray(eps_re_tm)
        eps_im_tm_gpu = cp.asarray(eps_im_tm)

        grid_size_tm = (
            (len(eps_re_tm) + block_size[0] - 1) // block_size[0],
            (len(eps_im_tm) + block_size[1] - 1) // block_size[1],
        )

        max_metric_per_a = np.zeros(len(radii_tm))
        best_eps_re_per_a = np.zeros(len(radii_tm))
        best_eps_im_per_a = np.zeros(len(radii_tm))

        out_tm_map = cp.zeros((len(eps_re_tm), len(eps_im_tm)), dtype=cp.float64)
        out_te_map_dummy = cp.zeros((len(eps_re_tm), len(eps_im_tm)), dtype=cp.float64)

        radii_iter_tm = (
            tqdm(radii_tm, desc="1D Optimization (TM) - GPU") if show_progress else radii_tm
        )
        with nvtx_range("1D TM optimization") if nvtx else nvtx_range(""):
            for i, a_val in enumerate(radii_iter_tm):
                with nvtx_range("TM kernel") if nvtx else nvtx_range(""):
                    kernel(
                        grid_size_tm,
                        block_size,
                        (
                            float(a_val),
                            eps_re_tm_gpu,
                            eps_im_tm_gpu,
                            wavelengths_gpu,
                            len(eps_re_tm),
                            len(eps_im_tm),
                            len(wavelengths),
                            n_max,
                            delta_wav,
                            out_tm_map,
                            out_te_map_dummy,
                        ),
                    )
                    cp.cuda.Stream.null.synchronize()
                max_idx = int(cp.argmax(out_tm_map))
                max_metric_per_a[i] = float(out_tm_map.ravel()[max_idx])

                re_idx = max_idx // len(eps_im_tm)
                im_idx = max_idx % len(eps_im_tm)
                best_eps_re_per_a[i] = eps_re_tm[re_idx]
                best_eps_im_per_a[i] = eps_im_tm[im_idx]

        results["tm"] = {
            "radii": radii_tm,
            "metric": max_metric_per_a,
            "eps_re": best_eps_re_per_a,
            "eps_im": best_eps_im_per_a,
        }
    else:
        results["tm"] = None

    if compute_te:
        eps_re_te_gpu = cp.asarray(eps_re_te)
        eps_im_te_gpu = cp.asarray(eps_im_te)

        grid_size_te = (
            (len(eps_re_te) + block_size[0] - 1) // block_size[0],
            (len(eps_im_te) + block_size[1] - 1) // block_size[1],
        )

        max_metric_per_a = np.zeros(len(radii_te))
        best_eps_re_per_a = np.zeros(len(radii_te))
        best_eps_im_per_a = np.zeros(len(radii_te))

        out_tm_map_dummy = cp.zeros((len(eps_re_te), len(eps_im_te)), dtype=cp.float64)
        out_te_map = cp.zeros((len(eps_re_te), len(eps_im_te)), dtype=cp.float64)

        radii_iter_te = (
            tqdm(radii_te, desc="1D Optimization (TE) - GPU") if show_progress else radii_te
        )
        with nvtx_range("1D TE optimization") if nvtx else nvtx_range(""):
            for i, a_val in enumerate(radii_iter_te):
                with nvtx_range("TE kernel") if nvtx else nvtx_range(""):
                    kernel(
                        grid_size_te,
                        block_size,
                        (
                            float(a_val),
                            eps_re_te_gpu,
                            eps_im_te_gpu,
                            wavelengths_gpu,
                            len(eps_re_te),
                            len(eps_im_te),
                            len(wavelengths),
                            n_max,
                            delta_wav,
                            out_tm_map_dummy,
                            out_te_map,
                        ),
                    )
                    cp.cuda.Stream.null.synchronize()
                max_idx = int(cp.argmax(out_te_map))
                max_metric_per_a[i] = float(out_te_map.ravel()[max_idx])

                re_idx = max_idx // len(eps_im_te)
                im_idx = max_idx % len(eps_im_te)
                best_eps_re_per_a[i] = eps_re_te[re_idx]
                best_eps_im_per_a[i] = eps_im_te[im_idx]

        results["te"] = {
            "radii": radii_te,
            "metric": max_metric_per_a,
            "eps_re": best_eps_re_per_a,
            "eps_im": best_eps_im_per_a,
        }
    else:
        results["te"] = None

    return results
