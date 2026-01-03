import numpy as np
import scipy.special as sp
import scipy.integrate as integrate
from tqdm import tqdm


def compute_metric(a, m_grid, wavelengths, n_max):
    n_arr = np.arange(n_max + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    wav = wavelengths[np.newaxis, :, np.newaxis, np.newaxis]
    m = m_grid[np.newaxis, np.newaxis, :, :]

    x = (2 * np.pi * a) / wav
    mx = m * x

    jn_x = sp.jv(n_arr, x)
    jn_prev_x = sp.jv(n_arr - 1, x)
    h2n_x = sp.hankel2(n_arr, x)
    h2n_prev_x = sp.hankel2(n_arr - 1, x)
    jn_mx = sp.jv(n_arr, mx)
    jn_prev_mx = sp.jv(n_arr - 1, mx)

    num_tm = (m * jn_x * jn_prev_mx) - (jn_mx * jn_prev_x)
    den_tm = (jn_mx * h2n_prev_x) - (m * h2n_x * jn_prev_mx)
    an_tm = num_tm / den_tm

    term_te = (n_arr / x) * (m - 1.0 / m)
    num_te = (jn_x * jn_prev_mx) - (m * jn_mx * jn_prev_x) + (term_te * jn_x * jn_mx)
    den_te = (m * jn_mx * h2n_prev_x) - (h2n_x * jn_prev_mx) - (term_te * jn_mx * h2n_x)
    an_te = num_te / den_te

    def sum_orders(an):
        q = np.real(an[0]) + np.abs(an[0]) ** 2
        q += 2 * np.sum(np.real(an[1:]) + np.abs(an[1:]) ** 2, axis=0)
        return -1 * (2.0 / x[0]) * q

    p_tm_spectral = sum_orders(an_tm)
    p_te_spectral = sum_orders(an_te)

    delta_wav = wavelengths[-1] - wavelengths[0]
    avg_tm = integrate.trapezoid(p_tm_spectral, wavelengths, axis=0) / delta_wav
    avg_te = integrate.trapezoid(p_te_spectral, wavelengths, axis=0) / delta_wav

    return avg_tm, avg_te


def run_cpu_sweep(
    wavelengths,
    radii_tm,
    radii_te,
    eps_re_tm,
    eps_im_tm,
    eps_re_te,
    eps_im_te,
    n_max,
    compute_tm=True,
    compute_te=True,
    show_progress=True,
):
    results = {}

    if compute_tm:
        re_mesh_tm, im_mesh_tm = np.meshgrid(eps_re_tm, eps_im_tm, indexing="ij")
        m_grid_tm = np.sqrt(re_mesh_tm + 1j * im_mesh_tm)

        best_tm_abs = np.full(re_mesh_tm.shape, -np.inf)
        best_tm_a = np.zeros(re_mesh_tm.shape)

        radii_iter_tm = (
            tqdm(radii_tm, desc="Sweeping Radii (TM) - CPU") if show_progress else radii_tm
        )
        for a in radii_iter_tm:
            tm_map, _ = compute_metric(a, m_grid_tm, wavelengths, n_max)
            mask_tm = tm_map > best_tm_abs
            best_tm_abs[mask_tm] = tm_map[mask_tm]
            best_tm_a[mask_tm] = a

        results["tm"] = {
            "best_abs": best_tm_abs,
            "best_a": best_tm_a,
            "re_mesh": re_mesh_tm,
            "im_mesh": im_mesh_tm,
        }
    else:
        results["tm"] = None

    if compute_te:
        re_mesh_te, im_mesh_te = np.meshgrid(eps_re_te, eps_im_te, indexing="ij")
        m_grid_te = np.sqrt(re_mesh_te + 1j * im_mesh_te)

        best_te_abs = np.full(re_mesh_te.shape, -np.inf)
        best_te_a = np.zeros(re_mesh_te.shape)

        radii_iter_te = (
            tqdm(radii_te, desc="Sweeping Radii (TE) - CPU") if show_progress else radii_te
        )
        for a in radii_iter_te:
            _, te_map = compute_metric(a, m_grid_te, wavelengths, n_max)
            mask_te = te_map > best_te_abs
            best_te_abs[mask_te] = te_map[mask_te]
            best_te_a[mask_te] = a

        results["te"] = {
            "best_abs": best_te_abs,
            "best_a": best_te_a,
            "re_mesh": re_mesh_te,
            "im_mesh": im_mesh_te,
        }
    else:
        results["te"] = None

    return results
