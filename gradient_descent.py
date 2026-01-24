from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpu_gradient import run_batched_gradient_descent


@dataclass
class OptimizationConfig:
    
    # Bounds for epsilon
    eps_re_min: float = -5.0
    eps_re_max: float = 10.0
    eps_im_min: float = -10.0
    eps_im_max: float = -0.05
    
    # Optimizer parameters (momentum SGD)
    learning_rate: float = 0.1
    momentum: float = 0.9
    lr_decay: float = 0.98
    max_iters: int = 80
    tolerance: float = 1e-8
    
    # Number of random restarts to avoid local minima
    n_restarts: int = 5


def optimize_eps_for_radii(
    radii: np.ndarray,
    wavelengths: np.ndarray,
    n_max: int,
    kernel_params: dict,
    config: OptimizationConfig,
    use_tm: bool = True,
) -> dict:

    return run_batched_gradient_descent(
        radii=radii,
        wavelengths=wavelengths,
        n_max=n_max,
        kernel_params=kernel_params,
        eps_re_min=config.eps_re_min,
        eps_re_max=config.eps_re_max,
        eps_im_min=config.eps_im_min,
        eps_im_max=config.eps_im_max,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        lr_decay=config.lr_decay,
        max_iters=config.max_iters,
        tolerance=config.tolerance,
        n_restarts=config.n_restarts,
        use_tm=use_tm,
    )


def run_gradient_descent_1d(
    wavelengths: np.ndarray,
    radii_tm: np.ndarray,
    radii_te: np.ndarray,
    n_max: int,
    kernel_params: dict,
    config_tm: OptimizationConfig | None = None,
    config_te: OptimizationConfig | None = None,
    compute_tm: bool = True,
    compute_te: bool = False,
) -> dict:

    results = {}
    
    if compute_tm:
        if config_tm is None:
            config_tm = OptimizationConfig()
        
        print(f"Running batched momentum SGD (TM): {len(radii_tm)} radii × {config_tm.n_restarts} restarts...")
        results["tm"] = optimize_eps_for_radii(
            radii=radii_tm,
            wavelengths=wavelengths,
            n_max=n_max,
            kernel_params=kernel_params,
            config=config_tm,
            use_tm=True,
        )
    else:
        results["tm"] = None
    
    if compute_te:
        if config_te is None:
            config_te = OptimizationConfig()
        
        print(f"Running batched momentum SGD (TE): {len(radii_te)} radii × {config_te.n_restarts} restarts...")
        results["te"] = optimize_eps_for_radii(
            radii=radii_te,
            wavelengths=wavelengths,
            n_max=n_max,
            kernel_params=kernel_params,
            config=config_te,
            use_tm=False,
        )
    else:
        results["te"] = None
    
    return results
