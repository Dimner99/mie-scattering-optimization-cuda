#include "mie_bessel_helpers.cuh"

extern "C" __global__ void mie_cylinder_grad_kernel(
    const double a,
    const double eps_re,
    const double eps_im,
    const double* wavelengths,
    int n_lambda,
    int n_max,
    double delta_wav,
    double* out_q_tm,
    double* out_q_te,
    double* out_grad_tm,
    double* out_grad_te)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    complex_t m = csqrt(make_complex(eps_re, eps_im));
    complex_t dm_deps_re = make_complex(0.5, 0.0) / m;
    complex_t dm_deps_im = make_complex(0.0, 0.5) / m;

    double q_tm_spectral[SPECTRAL_ARRAY_SIZE];
    double q_te_spectral[SPECTRAL_ARRAY_SIZE];
    double dq_tm_deps_re_spectral[SPECTRAL_ARRAY_SIZE];
    double dq_tm_deps_im_spectral[SPECTRAL_ARRAY_SIZE];
    double dq_tm_dr_spectral[SPECTRAL_ARRAY_SIZE];
    double dq_te_deps_re_spectral[SPECTRAL_ARRAY_SIZE];
    double dq_te_deps_im_spectral[SPECTRAL_ARRAY_SIZE];
    double dq_te_dr_spectral[SPECTRAL_ARRAY_SIZE];

    double jn_x_real[BESSEL_ARRAY_SIZE], yn_x_real[BESSEL_ARRAY_SIZE];
    complex_t jn_mx[BESSEL_ARRAY_SIZE], jn_x[BESSEL_ARRAY_SIZE], yn_x[BESSEL_ARRAY_SIZE], h2n_x[BESSEL_ARRAY_SIZE];

    for (int l = 0; l < n_lambda; ++l) {
        double wav = wavelengths[l];
        double x_val = (2.0 * M_PI * a) / wav;
        complex_t x = make_complex(x_val, 0.0);
        complex_t mx = m * x;
        double dx_dr = (2.0 * M_PI) / wav;

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
        double dq_tm_deps_re_lambda = 0.0;
        double dq_tm_deps_im_lambda = 0.0;
        double dq_tm_dr_lambda = 0.0;
        double dq_te_deps_re_lambda = 0.0;
        double dq_te_deps_im_lambda = 0.0;
        double dq_te_dr_lambda = 0.0;

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

            double dq_tm_deps_re_n = 0.0;
            double dq_tm_deps_im_n = 0.0;
            double dq_tm_dr_n = 0.0;
            double dq_te_deps_re_n = 0.0;
            double dq_te_deps_im_n = 0.0;
            double dq_te_dr_n = 0.0;

            dq_tm_deps_re_n = real((-(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x))*(-1.0/2.0*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*h2n_x_at((int)(n), h2n_x)/wav + (1.0/2.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n - 1), h2n_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))) + (-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))*((1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*jn_x_at((int)(n), jn_x)/wav - 1.0/2.0*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n - 1), jn_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))))*(2*conjc((cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x))/(-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))) + 1)/((-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x)) * (-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))));
dq_tm_deps_im_n = real((-(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x))*(-1.0/2.0*make_complex(0.0, 1.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*make_complex(0.0, 1.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*h2n_x_at((int)(n), h2n_x)/wav + (1.0/2.0)*make_complex(0.0, 1.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n - 1), h2n_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))) + (-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))*((1.0/2.0)*make_complex(0.0, 1.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*make_complex(0.0, 1.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*jn_x_at((int)(n), jn_x)/wav - 1.0/2.0*make_complex(0.0, 1.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n - 1), jn_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))))*(2*conjc((cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x))/(-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))) + 1)/((-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x)) * (-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))));
dq_tm_dr_n = real((-(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x))*(M_PI*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n - 1), h2n_x)/wav - M_PI*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*(h2n_x_at((int)(n - 1), h2n_x) - h2n_x_at((int)(n + 1), h2n_x))*jn_mx_at((int)(n - 1), jn_mx)/wav - M_PI*(make_complex(0.0, 1.0)*eps_im + eps_re)*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*h2n_x_at((int)(n), h2n_x)/wav + M_PI*(-h2n_x_at((int)(n), h2n_x) + h2n_x_at((int)(n - 2), h2n_x))*jn_mx_at((int)(n), jn_mx)/wav) + (-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))*(M_PI*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*(jn_x_at((int)(n - 1), jn_x) - jn_x_at((int)(n + 1), jn_x))*jn_mx_at((int)(n - 1), jn_mx)/wav - M_PI*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n - 1), jn_x)/wav + M_PI*(make_complex(0.0, 1.0)*eps_im + eps_re)*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*jn_x_at((int)(n), jn_x)/wav - M_PI*(-jn_x_at((int)(n), jn_x) + jn_x_at((int)(n - 2), jn_x))*jn_mx_at((int)(n), jn_mx)/wav))*(2*conjc((cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x))/(-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))) + 1)/((-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x)) * (-cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x) + jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x))));
            dq_te_deps_re_n = real((-((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) - cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x) + jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx))*(-1.0/2.0*wav*n*((1.0/2.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 3.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) - 1.0/4.0*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n - 1), h2n_x)/wav - 1.0/2.0*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*h2n_x_at((int)(n), h2n_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))) + (-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))*((1.0/2.0)*wav*n*((1.0/2.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 3.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) + (1.0/4.0)*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n), jn_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n - 1), jn_x)/wav + (1.0/2.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*jn_x_at((int)(n), jn_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))))*(2*conjc(((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) - cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x) + jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx))/(-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))) + 1)/((-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x)) * (-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))));
dq_te_deps_im_n = real((-((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) - cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x) + jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx))*(-1.0/2.0*wav*n*((1.0/2.0)*make_complex(0.0, 1.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*make_complex(0.0, 1.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 3.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) - 1.0/4.0*make_complex(0.0, 1.0)*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*make_complex(0.0, 1.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*make_complex(0.0, 1.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n - 1), h2n_x)/wav - 1.0/2.0*make_complex(0.0, 1.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*h2n_x_at((int)(n), h2n_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))) + (-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))*((1.0/2.0)*wav*n*((1.0/2.0)*make_complex(0.0, 1.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) + (1.0/2.0)*make_complex(0.0, 1.0)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 3.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) + (1.0/4.0)*make_complex(0.0, 1.0)*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n), jn_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*make_complex(0.0, 1.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x)/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1.0/2.0*make_complex(0.0, 1.0)*M_PI*a*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n - 1), jn_x)/wav + (1.0/2.0)*make_complex(0.0, 1.0)*M_PI*a*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*jn_x_at((int)(n), jn_x)/(wav*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))))*(2*conjc(((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) - cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x) + jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx))/(-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))) + 1)/((-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x)) * (-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))));
dq_te_dr_n = real((-((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) - cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x) + jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx))*((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*(a * a)) - 1.0/2.0*n*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n), h2n_x)/a - 1.0/2.0*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(h2n_x_at((int)(n - 1), h2n_x) - h2n_x_at((int)(n + 1), h2n_x))*jn_mx_at((int)(n), jn_mx)/a - M_PI*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*h2n_x_at((int)(n), h2n_x)/wav + M_PI*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*(-h2n_x_at((int)(n), h2n_x) + h2n_x_at((int)(n - 2), h2n_x))*jn_mx_at((int)(n), jn_mx)/wav + M_PI*(make_complex(0.0, 1.0)*eps_im + eps_re)*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*h2n_x_at((int)(n - 1), h2n_x)/wav - M_PI*(h2n_x_at((int)(n - 1), h2n_x) - h2n_x_at((int)(n + 1), h2n_x))*jn_mx_at((int)(n - 1), jn_mx)/wav) + (-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))*(-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*(a * a)) + (1.0/2.0)*n*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n), jn_x)/a + (1.0/2.0)*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*(jn_x_at((int)(n - 1), jn_x) - jn_x_at((int)(n + 1), jn_x))*jn_mx_at((int)(n), jn_mx)/a - M_PI*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*(-jn_x_at((int)(n), jn_x) + jn_x_at((int)(n - 2), jn_x))*jn_mx_at((int)(n), jn_mx)/wav + M_PI*cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*(-jn_mx_at((int)(n), jn_mx) + jn_mx_at((int)(n - 2), jn_mx))*jn_x_at((int)(n), jn_x)/wav - M_PI*(make_complex(0.0, 1.0)*eps_im + eps_re)*(jn_mx_at((int)(n - 1), jn_mx) - jn_mx_at((int)(n + 1), jn_mx))*jn_x_at((int)(n - 1), jn_x)/wav + M_PI*(jn_x_at((int)(n - 1), jn_x) - jn_x_at((int)(n + 1), jn_x))*jn_mx_at((int)(n - 1), jn_mx)/wav))*(2*conjc(((1.0/2.0)*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n), jn_mx)/(M_PI*a) - cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*jn_x_at((int)(n - 1), jn_x) + jn_x_at((int)(n), jn_x)*jn_mx_at((int)(n - 1), jn_mx))/(-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))) + 1)/((-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x)) * (-1.0/2.0*wav*n*(cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0) - 1/cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0))*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n), h2n_x)/(M_PI*a) + cpow(make_complex(0.0, 1.0)*eps_im + eps_re, 1.0/2.0)*jn_mx_at((int)(n), jn_mx)*h2n_x_at((int)(n - 1), h2n_x) - jn_mx_at((int)(n - 1), jn_mx)*h2n_x_at((int)(n), h2n_x))));

            dq_tm_deps_re_lambda += factor * dq_tm_deps_re_n;
            dq_tm_deps_im_lambda += factor * dq_tm_deps_im_n;
            dq_tm_dr_lambda += factor * dq_tm_dr_n;
            dq_te_deps_re_lambda += factor * dq_te_deps_re_n;
            dq_te_deps_im_lambda += factor * dq_te_deps_im_n;
            dq_te_dr_lambda += factor * dq_te_dr_n;
        }

        q_tm_spectral[l] = -1.0 * (2.0 / x_val) * q_tm_lambda;
        q_te_spectral[l] = -1.0 * (2.0 / x_val) * q_te_lambda;
        dq_tm_deps_re_spectral[l] = -1.0 * (2.0 / x_val) * dq_tm_deps_re_lambda;
        dq_tm_deps_im_spectral[l] = -1.0 * (2.0 / x_val) * dq_tm_deps_im_lambda;
        // For dr: dQ/dr = (2/x) × [dterm/dr - term/r] since Q = (-2/x)*term and x = 2πr/λ
        // The Wolfram expression dterm/dr is stored in dq_tm_dr_lambda
        // We need to subtract term/r contribution from the chain rule on 2/x
        dq_tm_dr_spectral[l] = -1.0 * (2.0 / x_val) * (dq_tm_dr_lambda - q_tm_lambda / a);
        dq_te_deps_re_spectral[l] = -1.0 * (2.0 / x_val) * dq_te_deps_re_lambda;
        dq_te_deps_im_spectral[l] = -1.0 * (2.0 / x_val) * dq_te_deps_im_lambda;
        dq_te_dr_spectral[l] = -1.0 * (2.0 / x_val) * (dq_te_dr_lambda - q_te_lambda / a);
    }

    double h = (wavelengths[n_lambda - 1] - wavelengths[0]) / (n_lambda - 1);
    double sum_q_tm = 0.5 * (q_tm_spectral[0] + q_tm_spectral[n_lambda - 1]);
    double sum_q_te = 0.5 * (q_te_spectral[0] + q_te_spectral[n_lambda - 1]);
    double sum_dq_tm_re = 0.5 * (dq_tm_deps_re_spectral[0] + dq_tm_deps_re_spectral[n_lambda - 1]);
    double sum_dq_tm_im = 0.5 * (dq_tm_deps_im_spectral[0] + dq_tm_deps_im_spectral[n_lambda - 1]);
    double sum_dq_tm_dr = 0.5 * (dq_tm_dr_spectral[0] + dq_tm_dr_spectral[n_lambda - 1]);
    double sum_dq_te_re = 0.5 * (dq_te_deps_re_spectral[0] + dq_te_deps_re_spectral[n_lambda - 1]);
    double sum_dq_te_im = 0.5 * (dq_te_deps_im_spectral[0] + dq_te_deps_im_spectral[n_lambda - 1]);
    double sum_dq_te_dr = 0.5 * (dq_te_dr_spectral[0] + dq_te_dr_spectral[n_lambda - 1]);

    for (int l = 1; l < n_lambda - 1; ++l) {
        sum_q_tm += q_tm_spectral[l];
        sum_q_te += q_te_spectral[l];
        sum_dq_tm_re += dq_tm_deps_re_spectral[l];
        sum_dq_tm_im += dq_tm_deps_im_spectral[l];
        sum_dq_tm_dr += dq_tm_dr_spectral[l];
        sum_dq_te_re += dq_te_deps_re_spectral[l];
        sum_dq_te_im += dq_te_deps_im_spectral[l];
        sum_dq_te_dr += dq_te_dr_spectral[l];
    }

    sum_q_tm *= h;
    sum_q_te *= h;
    sum_dq_tm_re *= h;
    sum_dq_tm_im *= h;
    sum_dq_tm_dr *= h;
    sum_dq_te_re *= h;
    sum_dq_te_im *= h;
    sum_dq_te_dr *= h;

    out_q_tm[0] = sum_q_tm / delta_wav;
    out_q_te[0] = sum_q_te / delta_wav;
    out_grad_tm[0] = sum_dq_tm_re / delta_wav;
    out_grad_tm[1] = sum_dq_tm_im / delta_wav;
    out_grad_tm[2] = sum_dq_tm_dr / delta_wav;
    out_grad_te[0] = sum_dq_te_re / delta_wav;
    out_grad_te[1] = sum_dq_te_im / delta_wav;
    out_grad_te[2] = sum_dq_te_dr / delta_wav;
}
