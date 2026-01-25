from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import time

from gpu_mie import run_gpu_1d_optimization
from gradient_descent import run_gradient_descent_1d, OptimizationConfig


@dataclass
class Config:
    
    # Wavelength range
    lambda_min: float = 400
    lambda_max: float = 800
    n_lambda: int = 400
    
    # Radii to sweep
    radii: np.ndarray = field(default_factory=lambda: np.linspace(4, 100, 96))
    
    
    # Epsilon bounds for optimization
    eps_re_min: float = -5.0
    eps_re_max: float = 10.0
    eps_im_min: float = -10.0
    eps_im_max: float = -0.05
    
    # Grid resolution for brute force
    n_eps_grid: int = 300
    
    # Number of terms in the series 
    n_max: int = 12
    
    # Modes to compute
    compute_tm: bool = True
    compute_te: bool = True


def build_kernel_params(n_max: int, n_lambda: int) -> dict:
    return {
        "miller_margin": 50,
        "miller_min_margin": 40,
        "miller_max_start": n_max + 50 + 30,
        "temp_array_size": n_max + 50 + 30 + 20,
        "bessel_array_size": n_max + 20,
        "spectral_array_size": n_lambda + 10,
        "series_max_iter": 60,
    }


def plot_comparison(brute_results: dict, grad_results: dict, mode: str = "tm"):
    brute = brute_results[mode]
    grad = grad_results[mode]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0, 0].plot(brute["radii"], brute["metric"], "b-", linewidth=2, label="Brute Force")
    axes[0, 0].plot(grad["radii"], grad["metric"], "r--", linewidth=2, label="Momentum SGD")
    axes[0, 0].set_xlabel("Radius a [nm]")
    axes[0, 0].set_ylabel("Max Q")
    axes[0, 0].set_title(f"{mode.upper()}: Spectral-Averaged Absorption")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(brute["radii"], brute["eps_re"], "b-", linewidth=2, label="Brute Force")
    axes[0, 1].plot(grad["radii"], grad["eps_re"], "r--", linewidth=2, label="Momentum SGD")
    axes[0, 1].set_xlabel("Radius a [nm]")
    axes[0, 1].set_ylabel("Optimal Re(ε)")
    axes[0, 1].set_title(f"{mode.upper()}: Optimal Re(ε) vs Radius")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(brute["radii"], brute["eps_im"], "b-", linewidth=2, label="Brute Force")
    axes[0, 2].plot(grad["radii"], grad["eps_im"], "r--", linewidth=2, label="Momentum SGD")
    axes[0, 2].set_xlabel("Radius a [nm]")
    axes[0, 2].set_ylabel("Optimal Im(ε)")
    axes[0, 2].set_title(f"{mode.upper()}: Optimal Im(ε) vs Radius")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    rel_diff_metric = (grad["metric"] - brute["metric"]) / (np.abs(brute["metric"]) + 1e-10) * 100
    axes[1, 0].plot(brute["radii"], rel_diff_metric, "g-", linewidth=2)
    axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("Radius a [nm]")
    axes[1, 0].set_ylabel("Relative Diff [%]")
    axes[1, 0].set_title("Q Difference: (Grad - Brute) / |Brute| × 100%")
    axes[1, 0].grid(True, alpha=0.3)
    
    diff_eps_re = grad["eps_re"] - brute["eps_re"]
    axes[1, 1].plot(brute["radii"], diff_eps_re, "g-", linewidth=2)
    axes[1, 1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Radius a [nm]")
    axes[1, 1].set_ylabel("Δ Re(ε)")
    axes[1, 1].set_title("Re(ε) Difference: Grad - Brute")
    axes[1, 1].grid(True, alpha=0.3)
    
    diff_eps_im = grad["eps_im"] - brute["eps_im"]
    axes[1, 2].plot(brute["radii"], diff_eps_im, "g-", linewidth=2)
    axes[1, 2].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 2].set_xlabel("Radius a [nm]")
    axes[1, 2].set_ylabel("Δ Im(ε)")
    axes[1, 2].set_title("Im(ε) Difference: Grad - Brute")
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"comparison_{mode}.png", dpi=150)
    plt.show()


def main():
    cfg = Config()
    wavelengths = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.n_lambda)
    kernel_params = build_kernel_params(cfg.n_max, cfg.n_lambda)
    
    eps_re_grid = np.linspace(cfg.eps_re_min, cfg.eps_re_max, cfg.n_eps_grid)
    eps_im_grid = np.linspace(cfg.eps_im_min, cfg.eps_im_max, cfg.n_eps_grid)
    
    print("=" * 60)
    print("Mie Cylinder Optimization")
    print("=" * 60)
    print(f"Wavelengths: {cfg.lambda_min}-{cfg.lambda_max} nm ({cfg.n_lambda} points)")
    print(f"Radii: {cfg.radii[0]:.1f}-{cfg.radii[-1]:.1f} nm ({len(cfg.radii)} points)")
    print(f"ε bounds: Re [{cfg.eps_re_min}, {cfg.eps_re_max}], Im [{cfg.eps_im_min}, {cfg.eps_im_max}]")
    print(f"Grid resolution: {cfg.n_eps_grid}×{cfg.n_eps_grid}")
    print("=" * 60)
    
    # Run brute force optimization
    print("\n[1/2] Running brute force grid search...")
    t0 = time.perf_counter()
    brute_results = run_gpu_1d_optimization(
        wavelengths=wavelengths,
        radii_tm=cfg.radii,
        radii_te=cfg.radii,
        eps_re_tm=eps_re_grid,
        eps_im_tm=eps_im_grid,
        eps_re_te=eps_re_grid,
        eps_im_te=eps_im_grid,
        n_max=cfg.n_max,
        kernel_params=kernel_params,
        compute_tm=cfg.compute_tm,
        compute_te=cfg.compute_te,
    )
    brute_elapsed = time.perf_counter() - t0
    print(f"Brute force time: {brute_elapsed:.2f}s")
    
    # Run gradient descent optimization
    print("\n[2/2] Running momentum SGD optimization...")
    opt_config = OptimizationConfig(
        eps_re_min=cfg.eps_re_min,
        eps_re_max=cfg.eps_re_max,
        eps_im_min=cfg.eps_im_min,
        eps_im_max=cfg.eps_im_max,
            learning_rate=0.1,
            momentum=0.9,
        lr_decay=0.98,
            max_iters=80,
        n_restarts=1,      
    )
    
    t0 = time.perf_counter()
    grad_results = run_gradient_descent_1d(
        wavelengths=wavelengths,
        radii_tm=cfg.radii,
        radii_te=cfg.radii,
        n_max=cfg.n_max,
        kernel_params=kernel_params,
        config_tm=opt_config,
        config_te=opt_config,
        compute_tm=cfg.compute_tm,
        compute_te=cfg.compute_te,
    )
    grad_elapsed = time.perf_counter() - t0
    print(f"Momentum SGD time: {grad_elapsed:.2f}s")
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    for mode in ["tm", "te"]:
        if brute_results.get(mode) is not None and grad_results.get(mode) is not None:
            brute = brute_results[mode]
            grad = grad_results[mode]
            
            brute_best_idx = np.argmax(brute["metric"])
            grad_best_idx = np.argmax(grad["metric"])
            
            print(f"\n{mode.upper()} Mode:")
            print(f"  Brute Force Best:")
            print(f"    Radius = {brute['radii'][brute_best_idx]:.2f} nm")
            print(f"    Q = {brute['metric'][brute_best_idx]:.6f}")
            print(f"    ε = {brute['eps_re'][brute_best_idx]:.3f} + {brute['eps_im'][brute_best_idx]:.3f}i")
            
            print(f"  Momentum SGD Best:")
            print(f"    Radius = {grad['radii'][grad_best_idx]:.2f} nm")
            print(f"    Q = {grad['metric'][grad_best_idx]:.6f}")
            print(f"    ε = {grad['eps_re'][grad_best_idx]:.3f} + {grad['eps_im'][grad_best_idx]:.3f}i")
            
            improvement = np.mean((grad["metric"] - brute["metric"]) / (np.abs(brute["metric"]) + 1e-10)) * 100
            print(f"  Average Q improvement: {improvement:+.2f}%")
    
    print("\nGenerating plots...")
    
    if cfg.compute_tm:
        plot_comparison(brute_results, grad_results, "tm")
    if cfg.compute_te:
        plot_comparison(brute_results, grad_results, "te")


if __name__ == "__main__":
    main()

