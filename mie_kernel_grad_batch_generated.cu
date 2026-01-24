/**
 * Batched momentum SGD kernel for Mie cylinder optimization.
 * 
 * Each thread handles one (radius, restart) pair and runs the full
 * Momentum SGD optimization internally.
 */

#include "mie_bessel_helpers.cuh"
#include "mie_optimizer_helpers.cuh"


__device__ void compute_q_and_grad_device(
    double a,
    double eps_re,
    double eps_im,
    const double* wavelengths,
    int n_lambda,
    int n_max,
    double delta_wav,
    bool use_tm,  // true = TM mode, false = TE mode
    double* out_q,
    double* out_grad_eps_re,
    double* out_grad_eps_im)
{
    complex_t m = csqrt(make_complex(eps_re, eps_im));

    double q_spectral[SPECTRAL_ARRAY_SIZE];
    double dq_deps_re_spectral[SPECTRAL_ARRAY_SIZE];
    double dq_deps_im_spectral[SPECTRAL_ARRAY_SIZE];

    double jn_x_real[BESSEL_ARRAY_SIZE], yn_x_real[BESSEL_ARRAY_SIZE];
    complex_t jn_mx[BESSEL_ARRAY_SIZE], jn_x[BESSEL_ARRAY_SIZE], yn_x[BESSEL_ARRAY_SIZE], h2n_x[BESSEL_ARRAY_SIZE];

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

        double q_lambda = 0.0;
        double dq_deps_re_lambda = 0.0;
        double dq_deps_im_lambda = 0.0;

        for (int n = 0; n <= n_max; ++n) {
            complex_t jn_prev_x = (n == 0) ? -jn_x[1] : jn_x[n - 1];
            complex_t jn_prev_mx = (n == 0) ? -jn_mx[1] : jn_mx[n - 1];
            complex_t h2n_prev_x = (n == 0) ? -h2n_x[1] : h2n_x[n - 1];

            double factor = (n == 0) ? 1.0 : 2.0;
            double dq_deps_re_n = 0.0;
            double dq_deps_im_n = 0.0;

            if (use_tm) {
                complex_t num_tm = m * jn_x[n] * jn_prev_mx - jn_mx[n] * jn_prev_x;
                complex_t den_tm = jn_mx[n] * h2n_prev_x - m * h2n_x[n] * jn_prev_mx;
                complex_t an_tm = num_tm / den_tm;
                q_lambda += factor * (real(an_tm) + abs2(an_tm));

                dq_deps_re_n = real((-(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x))*(-1.0/2.0*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*h2n_x_at((int)(n), h2n_x)/wav + (1.0/2.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n - 1), h2n_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))) + (-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))*((1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*jn_x_at((int)(n), jn_x)/wav - 1.0/2.0*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n - 1), jn_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))))*(2*conjc((cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x))/(-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))) + 1)/((-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x)) * (-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))));
                dq_deps_im_n = real((-(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x))*(-1.0/2.0*make_complex(0.0, 1.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*make_complex(0.0, 1.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*h2n_x_at((int)(n), h2n_x)/wav + (1.0/2.0)*make_complex(0.0, 1.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n - 1), h2n_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))) + (-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))*((1.0/2.0)*make_complex(0.0, 1.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*make_complex(0.0, 1.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*jn_x_at((int)(n), jn_x)/wav - 1.0/2.0*make_complex(0.0, 1.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n - 1), jn_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))))*(2*conjc((cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x))/(-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))) + 1)/((-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x)) * (-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))));
            } else {
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
                q_lambda += factor * (real(an_te) + abs2(an_te));

                dq_deps_re_n = real((-((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) - cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x) + jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx))*(-1.0/2.0*wav*n*((1.0/2.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 3.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) - 1.0/4.0*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n - 1), h2n_x)/wav - 1.0/2.0*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*h2n_x_at((int)(n), h2n_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))) + (-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))*((1.0/2.0)*wav*n*((1.0/2.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 3.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) + (1.0/4.0)*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n), jn_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n - 1), jn_x)/wav + (1.0/2.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*jn_x_at((int)(n), jn_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))))*(2*conjc(((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) - cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x) + jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx))/(-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))) + 1)/((-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x)) * (-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))));
                dq_deps_im_n = real((-((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) - cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x) + jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx))*(-1.0/2.0*wav*n*((1.0/2.0)*make_complex(0.0, 1.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*make_complex(0.0, 1.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 3.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) - 1.0/4.0*make_complex(0.0, 1.0)*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*make_complex(0.0, 1.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*make_complex(0.0, 1.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n - 1), h2n_x)/wav - 1.0/2.0*make_complex(0.0, 1.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*h2n_x_at((int)(n), h2n_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))) + (-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))*((1.0/2.0)*wav*n*((1.0/2.0)*make_complex(0.0, 1.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*make_complex(0.0, 1.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 3.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) + (1.0/4.0)*make_complex(0.0, 1.0)*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n), jn_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*make_complex(0.0, 1.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*make_complex(0.0, 1.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n - 1), jn_x)/wav + (1.0/2.0)*make_complex(0.0, 1.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*jn_x_at((int)(n), jn_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))))*(2*conjc(((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) - cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x) + jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx))/(-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))) + 1)/((-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x)) * (-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))));
            }

            dq_deps_re_lambda += factor * dq_deps_re_n;
            dq_deps_im_lambda += factor * dq_deps_im_n;
        }

        q_spectral[l] = -1.0 * (2.0 / x_val) * q_lambda;
        dq_deps_re_spectral[l] = -1.0 * (2.0 / x_val) * dq_deps_re_lambda;
        dq_deps_im_spectral[l] = -1.0 * (2.0 / x_val) * dq_deps_im_lambda;
    }

    // Trapezoidal integration
    double h = (wavelengths[n_lambda - 1] - wavelengths[0]) / (n_lambda - 1);
    double sum_q = 0.5 * (q_spectral[0] + q_spectral[n_lambda - 1]);
    double sum_dq_re = 0.5 * (dq_deps_re_spectral[0] + dq_deps_re_spectral[n_lambda - 1]);
    double sum_dq_im = 0.5 * (dq_deps_im_spectral[0] + dq_deps_im_spectral[n_lambda - 1]);

    for (int l = 1; l < n_lambda - 1; ++l) {
        sum_q += q_spectral[l];
        sum_dq_re += dq_deps_re_spectral[l];
        sum_dq_im += dq_deps_im_spectral[l];
    }

    sum_q *= h;
    sum_dq_re *= h;
    sum_dq_im *= h;

    *out_q = sum_q / delta_wav;
    *out_grad_eps_re = sum_dq_re / delta_wav;
    *out_grad_eps_im = sum_dq_im / delta_wav;
}

// ============================================================================
// Batched momentum SGD kernel
// Each thread optimizes one (radius, restart) pair
// ============================================================================

extern "C" __global__ void mie_gradient_descent_kernel(
    const double* radii,          // [n_radii]
    int n_radii,
    int n_restarts,
    const double* wavelengths,    // [n_lambda]
    int n_lambda,
    int n_max,
    double delta_wav,
    // Bounds
    double eps_re_min,
    double eps_re_max,
    double eps_im_min,
    double eps_im_max,
    // Optimization params
    double learning_rate,
    double momentum,
    double lr_decay,
    int max_iters,
    double tolerance,
    bool use_tm,                  // true = TM, false = TE
    // Random seeds for restarts (one per thread)
    const unsigned int* seeds,    // [n_radii * n_restarts]
    // Output: best result per radius (after reduction across restarts)
    double* out_best_q,           // [n_radii]
    double* out_best_eps_re,      // [n_radii]
    double* out_best_eps_im       // [n_radii]
)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = n_radii * n_restarts;
    if (global_idx >= total_threads) return;

    int radius_idx = global_idx / n_restarts;
    int restart_idx = global_idx % n_restarts;
    double a = radii[radius_idx];

    // Initialize eps based on restart index
    double eps_re, eps_im;
    if (restart_idx == 0) {
        // First restart: center of bounds
        eps_re = (eps_re_min + eps_re_max) / 2.0;
        eps_im = (eps_im_min + eps_im_max) / 2.0;
    } else {
        // Random restart
        unsigned int seed = seeds[global_idx];
        seed = seed * 1664525u + 1013904223u;
        eps_re = eps_re_min + (eps_re_max - eps_re_min) * (seed / 4294967296.0);
        seed = seed * 1664525u + 1013904223u;
        eps_im = eps_im_min + (eps_im_max - eps_im_min) * (seed / 4294967296.0);
    }

    double best_q = -1e30;
    double best_eps_re = eps_re;
    double best_eps_im = eps_im;
    
    double v_re = 0.0;
    double v_im = 0.0;
    double range_re = eps_re_max - eps_re_min;
    double range_im = eps_im_max - eps_im_min;
    double step_scale = fmin(range_re, range_im) * learning_rate;

    // Gradient ascent loop
    for (int iter = 0; iter < max_iters; ++iter) {
        double q_val, grad_re, grad_im;
        compute_q_and_grad_device(
            a, eps_re, eps_im,
            wavelengths, n_lambda, n_max, delta_wav,
            use_tm,
            &q_val, &grad_re, &grad_im
        );

        // Track best (only if valid)
        if (!isnan(q_val) && !isinf(q_val) && q_val > best_q) {
            best_q = q_val;
            best_eps_re = eps_re;
            best_eps_im = eps_im;
        }

        // Check for NaN gradients - skip this iteration
        if (isnan(grad_re) || isnan(grad_im) || isinf(grad_re) || isinf(grad_im)) {
            continue;
        }

        // Check convergence
        double grad_norm = sqrt(grad_re * grad_re + grad_im * grad_im);
        if (grad_norm < tolerance) break;
        
        double grad_re_unit = grad_re / grad_norm;
        double grad_im_unit = grad_im / grad_norm;

        double step_re = 0.0;
        double step_im = 0.0;
        momentum_update(grad_re_unit, momentum, step_scale, &v_re, &step_re);
        momentum_update(grad_im_unit, momentum, step_scale, &v_im, &step_im);
        
        double new_re = eps_re + step_re;
        double new_im = eps_im + step_im;
        
        // Clip to bounds
        new_re = fmax(eps_re_min, fmin(eps_re_max, new_re));
        new_im = fmax(eps_im_min, fmin(eps_im_max, new_im));
        
        eps_re = new_re;
        eps_im = new_im;
        step_scale *= lr_decay;
    }


    int out_idx = global_idx;  
    
    __shared__ double shared_q[256];
    __shared__ double shared_re[256];
    __shared__ double shared_im[256];
    __shared__ int shared_radius_idx[256];

    int tid = threadIdx.x;
    shared_q[tid] = best_q;
    shared_re[tid] = best_eps_re;
    shared_im[tid] = best_eps_im;
    shared_radius_idx[tid] = radius_idx;
    __syncthreads();

    out_best_q[global_idx] = best_q;
    out_best_eps_re[global_idx] = best_eps_re;
    out_best_eps_im[global_idx] = best_eps_im;
}
