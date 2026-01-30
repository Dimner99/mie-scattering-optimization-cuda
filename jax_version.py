#%%
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import matplotlib.pyplot as plt
import time

jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 0. Configuration
# ==============================================================================

@dataclass(frozen=True)
class OptimizationConfig:
    eps_re_min: float
    eps_re_max: float 
    eps_im_min: float
    eps_im_max: float
    learning_rate: float
    momentum: float 
    lr_decay: float 
    max_iters: int
    n_restarts: int
    mode: str = "tm"       # "tm", "te"
    use_checkpoint: bool = False  

# ==============================================================================
# 1. Pure-JAX Bessel Solvers (no CPU callbacks)
# ==============================================================================

# -- Adaptive Miller margin: how many extra backward steps beyond n_max --
MILLER_MARGIN = 50
MILLER_MIN_MARGIN = 40
# Maximum total backward start index
MILLER_MAX_START = 92

def bessel_j0_series(z, max_iter=60):
    """Power-series expansion for J0(z), works for real or complex z."""
    z_half = z / 2.0
    z_half_sq = z_half * z_half
    term = jnp.ones_like(z)
    total = jnp.ones_like(z)

    def body(k, state):
        term_val, total_val = state
        term_val = term_val * (-z_half_sq) / (k * k)
        total_val = total_val + term_val
        return term_val, total_val

    term, total = lax.fori_loop(1, max_iter + 1, body, (term, total))
    return total


def bessel_j1_series(z, max_iter=60):
    """Power-series expansion for J1(z), works for real or complex z."""
    z_half = z / 2.0
    z_half_sq = z_half * z_half
    # First term: (z/2)^1 / 1! = z/2
    term = z_half
    total = z_half

    def body(k, state):
        term_val, total_val = state
        # term_k = term_{k-1} * (-z_half_sq) / (k * (1 + k))
        term_val = term_val * (-z_half_sq) / (k * (1 + k))
        total_val = total_val + term_val
        return term_val, total_val

    term, total = lax.fori_loop(1, max_iter + 1, body, (term, total))
    return total


def bessel_y0_series(x, max_iter=40):
    """
    Logarithmic series for Y0(x) — pure JAX, no scipy callback.
    Y0(x) = (2/pi) * [(ln(x/2) + gamma) * J0(x) + sum_{k>=1} ...]
    Matches the CUDA bessel_y_series(0, z) implementation.
    """
    gamma_const = 0.5772156649015329
    x_half = x / 2.0
    x_half_sq = x_half * x_half
    ln_x_half = jnp.log(x_half)

    j0_val = bessel_j0_series(x, max_iter=60)

    # Correction sum: sum_{k=1}^{max_iter} (-x_half_sq)^k / (k!)^2 * H_k
    term = jnp.ones_like(x)  # (x/2)^0 = 1
    harmonic = jnp.zeros_like(x)
    correction = jnp.zeros_like(x)

    def body(k, state):
        term_val, harmonic_val, corr_val = state
        term_val = term_val * (-x_half_sq) / (k * k)
        harmonic_val = harmonic_val + 1.0 / k
        corr_val = corr_val + term_val * harmonic_val
        return term_val, harmonic_val, corr_val

    _, _, correction = lax.fori_loop(1, max_iter + 1, body, (term, harmonic, correction))

    return (2.0 / jnp.pi) * ((ln_x_half + gamma_const) * j0_val - correction)


def bessel_y1_wronskian(x):
    """
    Compute Y1(x) from Y0(x) using the Wronskian identity:
        J0(x)*Y1(x) - J1(x)*Y0(x) = -2/(pi*x)
    Solving: Y1(x) = [J1(x)*Y0(x) - 2/(pi*x)] / J0(x)
    """
    j0 = bessel_j0_series(x, max_iter=60)
    j1 = bessel_j1_series(x, max_iter=60)
    y0 = bessel_y0_series(x, max_iter=40)
    return (j1 * y0 - 2.0 / (jnp.pi * x)) / j0


def miller_bessel_j_sequence_real(x, n_max):
    """
    Backward recurrence for real x with normalization by J0 series.
    Adaptive start index: n_start = min(MILLER_MAX_START, max(n_max + MILLER_MIN_MARGIN, n_max + floor(x) + MILLER_MARGIN))
    Since lax.scan requires static shapes, we always allocate MILLER_MAX_START
    steps but mask unused entries to zero, which the normalization absorbs.
    """
    n_start = MILLER_MAX_START  # fixed static shape for scan

    def backward_step(carry, n):
        jn, jn_plus_1 = carry
        n_c = n.astype(x.dtype)
        jn_minus_1 = (2.0 * n_c / x) * jn - jn_plus_1
        return (jn_minus_1, jn), jn_minus_1

    init_val = (jnp.array(1e-30, dtype=x.dtype), jnp.array(0.0, dtype=x.dtype))
    scan_indices = jnp.arange(n_start, 0, -1)
    _, seq_reversed = lax.scan(backward_step, init_val, scan_indices)
    full_seq = seq_reversed[::-1][: n_max + 1]


    j0 = bessel_j0_series(jnp.asarray(x, dtype=jnp.float64), max_iter=60)
    scale = j0 / full_seq[0]
    return full_seq * scale


def miller_bessel_j_sequence_complex(z, n_max):
    """Backward recurrence for complex z with normalization by series J0(z)."""
    n_start = MILLER_MAX_START

    def backward_step(carry, n):
        jn, jn_plus_1 = carry
        n_c = n.astype(z.dtype)
        jn_minus_1 = (2.0 * n_c / z) * jn - jn_plus_1
        return (jn_minus_1, jn), jn_minus_1

    init_val = (jnp.array(1e-30, dtype=z.dtype), jnp.array(0.0, dtype=z.dtype))
    scan_indices = jnp.arange(n_start, 0, -1)
    _, seq_reversed = lax.scan(backward_step, init_val, scan_indices)
    full_seq = seq_reversed[::-1][: n_max + 1]

    j0 = bessel_j0_series(z, max_iter=60)
    scale = j0 / full_seq[0]
    return full_seq * scale


def forward_bessel_y_sequence(x, n_max):
    """
    Computes [Y_0(x), ..., Y_n_max(x)] using forward recurrence.
    """
    y0 = bessel_y0_series(x, max_iter=40)
    y1 = bessel_y1_wronskian(x)

    def forward_step(carry, n):
        yn_minus_1, yn = carry
        n_c = n.astype(x.dtype)
        yn_plus_1 = (2.0 * n_c / x) * yn - yn_minus_1
        return (yn, yn_plus_1), yn_plus_1

    scan_indices = jnp.arange(1, n_max)
    _, y_tail = lax.scan(forward_step, (y0, y1), scan_indices)

    return jnp.concatenate([y0[None], y1[None], y_tail], axis=0)


def complex_hankel_h2_seq(j_seq, y_seq):
    """H_n^(2) = J_n - i * Y_n"""
    return j_seq - 1j * y_seq

# ==============================================================================
# 2. Physics Engine — computes both TM and TE per wavelength
# ==============================================================================

def compute_q_both_vectorized(eps_re, eps_im, radius, wavelength, n_max):
    """
    Compute Q_abs for TM and TE modes at a single wavelength.
    Returns (q_tm, q_te).
    """
    m = jnp.sqrt(eps_re + 1j * eps_im)
    x = 2.0 * jnp.pi * radius / wavelength
    mx = m * x

    # Bessel sequences (shared between TM and TE)
    jn_mx_seq = miller_bessel_j_sequence_complex(mx, n_max)
    jn_x_seq = miller_bessel_j_sequence_real(x, n_max)
    yn_x_seq = forward_bessel_y_sequence(x, n_max)
    h2n_x_seq = complex_hankel_h2_seq(jn_x_seq, yn_x_seq)

    # Previous-order arrays using J_{-1} = -J_1 convention
    jn_prev_x = jnp.concatenate([-jn_x_seq[1:2], jn_x_seq[:-1]])
    jn_prev_mx = jnp.concatenate([-jn_mx_seq[1:2], jn_mx_seq[:-1]])
    h2n_prev_x = jnp.concatenate([-h2n_x_seq[1:2], h2n_x_seq[:-1]])

    n_idx = jnp.arange(n_max + 1)
    factor = jnp.where(n_idx == 0, 1.0, 2.0)

    # --- TM coefficients ---
    num_tm = m * jn_x_seq * jn_prev_mx - jn_mx_seq * jn_prev_x
    den_tm = jn_mx_seq * h2n_prev_x - m * h2n_x_seq * jn_prev_mx
    an_tm = num_tm / den_tm

    term_tm = jnp.real(an_tm) + jnp.abs(an_tm) ** 2
    q_tm = -(2.0 / x) * jnp.sum(factor * term_tm)

    # --- TE coefficients ---
    m_inv = 1.0 / m
    te_extra = (n_idx / x) * (m - m_inv)
    num_te = (jn_x_seq * jn_prev_mx - m * jn_mx_seq * jn_prev_x
              + te_extra * jn_x_seq * jn_mx_seq)
    den_te = (m * jn_mx_seq * h2n_prev_x - h2n_x_seq * jn_prev_mx
              - te_extra * jn_mx_seq * h2n_x_seq)
    an_te = num_te / den_te

    term_te = jnp.real(an_te) + jnp.abs(an_te) ** 2
    q_te = -(2.0 / x) * jnp.sum(factor * term_te)

    return q_tm, q_te


def spectral_q_objective(params, radius, wavelengths, n_max, mode):
    """
    Spectral-averaged Q_abs using trapezoidal integration.
    mode: "tm" or "te"
    """
    eps_re, eps_im = params[0], params[1]
    batch_fn = vmap(compute_q_both_vectorized, in_axes=(None, None, None, 0, None))
    q_tm_vals, q_te_vals = batch_fn(eps_re, eps_im, radius, wavelengths, n_max)

    delta_wav = wavelengths[-1] - wavelengths[0]

    def trapezoid(vals):
        return jnp.trapezoid(vals, wavelengths) / delta_wav

    if mode == "tm":
        return trapezoid(q_tm_vals)
    else:
        return trapezoid(q_te_vals)

# ==============================================================================
# 3. Optimization
# ==============================================================================

def optimize_single_radius(radius, restart_key, wavelengths, n_max, config):
    k1, k2 = jax.random.split(restart_key)
    re = jax.random.uniform(k1, minval=config.eps_re_min, maxval=config.eps_re_max)
    im = jax.random.uniform(k2, minval=config.eps_im_min, maxval=config.eps_im_max)
    params = jnp.array([re, im])
    velocity = jnp.zeros_like(params)
    step_scale = jnp.minimum(
        config.eps_re_max - config.eps_re_min,
        config.eps_im_max - config.eps_im_min,
    ) * config.learning_rate

    # Best-iterate tracking
    init_q = spectral_q_objective(params, radius, wavelengths, n_max, config.mode)
    best_p = params
    best_q = init_q

    def raw_step(carry, _):
        p, v, scale, b_p, b_q = carry
        q_val, grads = jax.value_and_grad(spectral_q_objective)(
            p, radius, wavelengths, n_max, config.mode
        )
        grad_norm = jnp.linalg.norm(grads)
        grad_unit = grads / (grad_norm + 1e-12)
        v = config.momentum * v + (1.0 - config.momentum) * grad_unit
        new_p = p + scale * v

        new_re = jnp.clip(new_p[0], config.eps_re_min, config.eps_re_max)
        new_im = jnp.clip(new_p[1], config.eps_im_min, config.eps_im_max)
        new_p = jnp.array([new_re, new_im])

        scale = scale * config.lr_decay

        improved = q_val > b_q
        b_p = jnp.where(improved, p, b_p)
        b_q = jnp.where(improved, q_val, b_q)

        return (new_p, v, scale, b_p, b_q), q_val

    step_fn = jax.checkpoint(raw_step) if config.use_checkpoint else raw_step

    init_carry = (params, velocity, step_scale, best_p, best_q)
    final, q_hist = lax.scan(step_fn, init_carry, jnp.arange(config.max_iters))
    _, _, _, best_p_final, best_q_final = final

    last_q = q_hist[-1]
    final_p = final[0]
    use_last = last_q > best_q_final
    best_p_final = jnp.where(use_last, final_p, best_p_final)
    best_q_final = jnp.where(use_last, last_q, best_q_final)

    return best_p_final, best_q_final


@partial(jit, static_argnums=(2, 3))
def run_jax_optimization(radii, wavelengths, n_max, config, rng_key):
    n_radii = len(radii)
    n_restarts = config.n_restarts 
    
    keys = jax.random.split(rng_key, n_radii * n_restarts)
    keys = keys.reshape(n_radii, n_restarts, 2)
    radii_exp = jnp.repeat(radii[:, None], n_restarts, axis=1)
    
    def opt_func(r, k):
        return optimize_single_radius(r, k, wavelengths, n_max, config)
    
    batch_opt = vmap(vmap(opt_func, in_axes=(0, 0)), in_axes=(0, 0))
    all_params, all_qs = batch_opt(radii_exp, keys)
    
    best_idx = jnp.argmax(all_qs, axis=1)
    row_idx = jnp.arange(n_radii)
    
    return {
        "radii": radii,
        "metric": all_qs[row_idx, best_idx],
        "eps_re": all_params[row_idx, best_idx, 0],
        "eps_im": all_params[row_idx, best_idx, 1]
    }

# ==============================================================================
# 4. Main
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JAX Mie Cylinder Optimizer")
    parser.add_argument("--mode", choices=["tm", "te", "both"], default="tm",
                        help="Polarization mode to optimise (default: tm)")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Enable gradient checkpointing (saves memory, slower)")
    parser.add_argument("--n-restarts", type=int, default=1,
                        help="Random restarts per radius (default: 1)")
    parser.add_argument("--max-iters", type=int, default=80,
                        help="Optimizer iterations per restart (default: 80)")
    parser.add_argument("--n-radii", type=int, default=96,
                        help="Number of radii to sweep (default: 96)")
    parser.add_argument("--n-lambda", type=int, default=400,
                        help="Number of wavelength points (default: 400)")
    args = parser.parse_args()

    config = OptimizationConfig(
        eps_re_min=-5.0, eps_re_max=10.0,
        eps_im_min=-10.0, eps_im_max=-0.05,
        learning_rate=0.1,
        momentum=0.9,
        lr_decay=0.98,
        max_iters=args.max_iters,
        n_restarts=args.n_restarts,
        mode=args.mode,
        use_checkpoint=args.checkpoint,
    )
    
    radii = jnp.linspace(4, 100, args.n_radii)
    wavelengths = jnp.linspace(400, 800, args.n_lambda)
    n_max = 12
    key = jax.random.PRNGKey(42)
    
    print(f"Mode={config.mode}  MomentumSGD(lr={config.learning_rate}, momentum={config.momentum}, decay={config.lr_decay})  "
          f"iters={config.max_iters}  restarts={config.n_restarts}  "
          f"checkpoint={config.use_checkpoint}")
    print(f"Radii: {args.n_radii} pts [{float(radii[0]):.0f}–{float(radii[-1]):.0f} nm]  "
          f"Wavelengths: {args.n_lambda} pts")
    print("Compiling JAX kernel (pure-JAX Bessel, momentum SGD optimizer)...")
    t0 = time.perf_counter()
    results = run_jax_optimization(radii, wavelengths, n_max, config, key)
    jax.block_until_ready(results["metric"])
    total_time = time.perf_counter() - t0
    print(f"Finished in {total_time:.4f}s")
    
    print("Lets measure the cached version time.")
    t0 = time.perf_counter()
    results = run_jax_optimization(radii, wavelengths, n_max, config, key)
    jax.block_until_ready(results["metric"])
    total_time = time.perf_counter() - t0
    print(f"Finished in {total_time:.4f}s")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(results['radii'], results['metric'], 'b-')
    axes[0].grid(True, alpha=0.3, which='both', linestyle='--')
    axes[0].set_title(f"Optimal Q ({config.mode.upper()})")
    axes[0].set_xlabel("Radius [nm]"); axes[0].set_ylabel("Q_abs")
    axes[1].plot(results['radii'], results['eps_re'], 'r-')
    axes[1].set_title("Re(ε)"); axes[1].set_xlabel("Radius [nm]")
    axes[2].plot(results['radii'], results['eps_im'], 'g-')
    axes[2].set_title("Im(ε)"); axes[2].set_xlabel("Radius [nm]")
    plt.tight_layout()
    plt.savefig(f"jax_results_{config.mode}.png", dpi=150)
    print(f"Plot saved to jax_results_{config.mode}.png")
    plt.show()