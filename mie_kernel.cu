#include "mie_bessel_helpers.cuh"


extern "C" __global__ void mie_cylinder_kernel(
    const double a, const double* eps_re_grid, const double* eps_im_grid,
    const double* wavelengths, int n_re, int n_im, int n_lambda, int n_max,
    double delta_wav,
    double* out_tm_map, double* out_te_map)
{
    int re_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int im_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (re_idx >= n_re || im_idx >= n_im) return;

    complex_t m = csqrt(make_complex(eps_re_grid[re_idx], eps_im_grid[im_idx]));

    double sum_p_tm = 0.0;
    double sum_p_te = 0.0;

    double jn_x_real[BESSEL_ARRAY_SIZE], yn_x_real[BESSEL_ARRAY_SIZE];
    complex_t jn_mx[BESSEL_ARRAY_SIZE], jn_x[BESSEL_ARRAY_SIZE], yn_x[BESSEL_ARRAY_SIZE], h2n_x[BESSEL_ARRAY_SIZE];

    double p_tm_spectral[SPECTRAL_ARRAY_SIZE];
    double p_te_spectral[SPECTRAL_ARRAY_SIZE];

    for (int l = 0; l < n_lambda; ++l) {
        double wav = wavelengths[l];
        double x_val = (2.0 * M_PI * a) / wav;
        complex_t x = make_complex(x_val, 0.0);
        complex_t mx = m * x;

        bessel_j_real_miller(x_val, n_max, jn_x_real);
        bessel_y_real(x_val, n_max, yn_x_real);

        for (int n = 0; n <= n_max; n++) {
            jn_x[n] = make_complex(jn_x_real[n], 0.0);
            yn_x[n] = make_complex(yn_x_real[n], 0.0);
            h2n_x[n] = jn_x[n] - make_complex(0.0, 1.0) * yn_x[n];
        }

        bessel_j_complex_miller(mx, n_max, jn_mx);

        double q_tm_lambda = 0.0;
        double q_te_lambda = 0.0;

        for (int n = 0; n <= n_max; ++n) {
            complex_t jn_prev_x = (n == 0) ? -jn_x[1] : jn_x[n - 1];
            complex_t jn_prev_mx = (n == 0) ? -jn_mx[1] : jn_mx[n - 1];
            complex_t h2n_prev_x = (n == 0) ? -h2n_x[1] : h2n_x[n - 1];

            complex_t num_tm = m * jn_x[n] * jn_prev_mx - jn_mx[n] * jn_prev_x;
            complex_t den_tm = jn_mx[n] * h2n_prev_x - m * h2n_x[n] * jn_prev_mx;
            complex_t an_tm = num_tm / den_tm;

            complex_t num_te, den_te;
            if (x_val > 1e-9 && n > 0) {
                complex_t m_inv = make_complex(1.0, 0.0) / m;
                complex_t term_te = make_complex((double)n / x_val, 0.0) * (m - m_inv);
                num_te = (jn_x[n] * jn_prev_mx - m * jn_mx[n] * jn_prev_x) + term_te * jn_x[n] * jn_mx[n];
                den_te = (m * jn_mx[n] * h2n_prev_x - h2n_x[n] * jn_prev_mx) - term_te * jn_mx[n] * h2n_x[n];
            } else {
                num_te = jn_x[n] * jn_prev_mx - m * jn_mx[n] * jn_prev_x;
                den_te = m * jn_mx[n] * h2n_prev_x - h2n_x[n] * jn_prev_mx;
            }
            complex_t an_te = num_te / den_te;

            double factor = (n == 0) ? 1.0 : 2.0;
            q_tm_lambda += factor * (real(an_tm) + abs2(an_tm));
            q_te_lambda += factor * (real(an_te) + abs2(an_te));
        }
        p_tm_spectral[l] = -1.0 * (2.0 / x_val) * q_tm_lambda;
        p_te_spectral[l] = -1.0 * (2.0 / x_val) * q_te_lambda;
    }

    double h = (wavelengths[n_lambda - 1] - wavelengths[0]) / (n_lambda - 1);
    sum_p_tm = 0.5 * (p_tm_spectral[0] + p_tm_spectral[n_lambda - 1]);
    sum_p_te = 0.5 * (p_te_spectral[0] + p_te_spectral[n_lambda - 1]);
    for (int l = 1; l < n_lambda - 1; ++l) {
        sum_p_tm += p_tm_spectral[l];
        sum_p_te += p_te_spectral[l];
    }
    sum_p_tm *= h;
    sum_p_te *= h;

    int idx = re_idx * n_im + im_idx;
    out_tm_map[idx] = sum_p_tm / delta_wav;
    out_te_map[idx] = sum_p_te / delta_wav;
}
