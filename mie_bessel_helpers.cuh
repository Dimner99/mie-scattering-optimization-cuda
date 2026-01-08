#ifndef MIE_BESSEL_HELPERS_CUH
#define MIE_BESSEL_HELPERS_CUH

#include <cuComplex.h>

typedef cuDoubleComplex complex_t;

__device__ inline complex_t make_complex(double r, double i) { return make_cuDoubleComplex(r, i); }
__device__ inline double real(const complex_t& c) { return cuCreal(c); }
__device__ inline double real(double x) { return x; }
__device__ inline double imag(const complex_t& c) { return cuCimag(c); }
__device__ inline double abs2(const complex_t& c) { return real(c) * real(c) + imag(c) * imag(c); }
__device__ inline double cabs(const complex_t& c) { return sqrt(abs2(c)); }
__device__ inline complex_t conjc(const complex_t& c) { return make_complex(real(c), -imag(c)); }
__device__ inline complex_t cneg(const complex_t& c) { return make_complex(-real(c), -imag(c)); }

// Handling the negative index of Bessel
// J_{-n}(x) = (-1)^n * J_n(x), H2_{-n}(x) = (-1)^n * H2_n(x)
__device__ inline complex_t jn_x_at(int idx, const complex_t* jn_x) {
    if (idx >= 0) return jn_x[idx];
    int n = -idx;
    return (n % 2 == 0) ? jn_x[n] : cneg(jn_x[n]);
}
__device__ inline complex_t jn_mx_at(int idx, const complex_t* jn_mx) {
    if (idx >= 0) return jn_mx[idx];
    int n = -idx;
    return (n % 2 == 0) ? jn_mx[n] : cneg(jn_mx[n]);
}
__device__ inline complex_t h2n_x_at(int idx, const complex_t* h2n_x) {
    if (idx >= 0) return h2n_x[idx];
    int n = -idx;
    return (n % 2 == 0) ? h2n_x[n] : cneg(h2n_x[n]);
}

__device__ inline complex_t csqrt(const complex_t& z) {
    double a = real(z);
    double b = imag(z);
    double r = cabs(z);
    double re = sqrt((r + a) / 2.0);
    double im = (b >= 0 ? 1.0 : -1.0) * sqrt((r - a) / 2.0);
    return make_complex(re, im);
}

__device__ inline complex_t cexp(const complex_t& z) {
    double ea = exp(real(z));
    return make_complex(ea * cos(imag(z)), ea * sin(imag(z)));
}

__device__ inline complex_t clog(const complex_t& z) {
    return make_complex(log(cabs(z)), atan2(imag(z), real(z)));
}

__device__ inline complex_t cpow(const complex_t& z, double w) {
    complex_t logz = clog(z);
    return cexp(make_complex(w * real(logz), w * imag(logz)));
}

__device__ inline complex_t operator+(const complex_t& a, const complex_t& b) { return cuCadd(a, b); }
__device__ inline complex_t operator-(const complex_t& a, const complex_t& b) { return cuCsub(a, b); }
__device__ inline complex_t operator*(const complex_t& a, const complex_t& b) { return cuCmul(a, b); }
__device__ inline complex_t operator/(const complex_t& a, const complex_t& b) { return cuCdiv(a, b); }
__device__ inline complex_t operator-(const complex_t& a) { return make_complex(-real(a), -imag(a)); }

__device__ inline complex_t operator*(double a, const complex_t& b) { return make_complex(a * real(b), a * imag(b)); }
__device__ inline complex_t operator*(const complex_t& a, double b) { return make_complex(real(a) * b, imag(a) * b); }
__device__ inline complex_t operator/(const complex_t& a, double b) { return make_complex(real(a) / b, imag(a) / b); }
__device__ inline complex_t operator/(double a, const complex_t& b) {
    double denom = real(b) * real(b) + imag(b) * imag(b);
    return make_complex(a * real(b) / denom, -a * imag(b) / denom);
}
__device__ inline complex_t operator+(const complex_t& a, double b) { return make_complex(real(a) + b, imag(a)); }
__device__ inline complex_t operator+(double a, const complex_t& b) { return make_complex(a + real(b), imag(b)); }
__device__ inline complex_t operator-(const complex_t& a, double b) { return make_complex(real(a) - b, imag(a)); }
__device__ inline complex_t operator-(double a, const complex_t& b) { return make_complex(a - real(b), -imag(b)); }

__device__ complex_t bessel_j_series(int n, const complex_t& z) {
    complex_t z_half = z / 2.0;
    complex_t z_half_sq = z_half * z_half;

    complex_t prefix = make_complex(1.0, 0.0);
    for (int i = 0; i < n; i++) {
        prefix = prefix * z_half;
    }

    complex_t sum = make_complex(1.0, 0.0);
    complex_t term = make_complex(1.0, 0.0);

    for (int k = 1; k <= SERIES_MAX_ITER; k++) {
        term = term * (-z_half_sq) / (double)(k * (n + k));
        sum = sum + term;
        if (abs2(term) < 1e-30 * abs2(sum)) break;
    }

    double nfact = 1.0;
    for (int i = 2; i <= n; i++) nfact *= i;

    return prefix * sum / nfact;
}

__device__ complex_t bessel_y_series(int n, const complex_t& z) {
    double gamma_const = 0.5772156649015329;
    complex_t z_half = z / 2.0;
    complex_t ln_z_half = clog(z_half);

    if (n == 0) {
        complex_t j0 = bessel_j_series(0, z);
        complex_t sum = make_complex(0.0, 0.0);
        complex_t z_half_sq = z_half * z_half;
        complex_t term = make_complex(1.0, 0.0);
        double harmonic = 0.0;

        for (int k = 1; k <= 40; k++) {
            term = term * (-z_half_sq) / (double)(k * k);
            harmonic += 1.0 / k;
            sum = sum + term * harmonic;
            if (abs2(term) < 1e-30) break;
        }

        return (2.0 / M_PI) * ((ln_z_half + gamma_const) * j0 + sum);
    } else if (n == 1) {
        complex_t j1 = bessel_j_series(1, z);
        complex_t sum = make_complex(-1.0, 0.0);
        complex_t z_half_sq = z_half * z_half;
        complex_t term = z_half;
        double harmonic = 1.0;

        for (int k = 1; k <= 40; k++) {
            term = term * (-z_half_sq) / (double)(k * (k + 1));
            harmonic += 1.0 / k + 1.0 / (k + 1);
            sum = sum + term * harmonic;
            if (abs2(term) < 1e-30) break;
        }

        return (2.0 / M_PI) * ((ln_z_half + gamma_const) * j1 - 1.0 / z + sum);
    } else {
        complex_t y_prev = bessel_y_series(0, z);
        complex_t y_curr = bessel_y_series(1, z);

        for (int k = 1; k < n; k++) {
            complex_t y_next = (2.0 * k / z) * y_curr - y_prev;
            y_prev = y_curr;
            y_curr = y_next;
        }
        return y_curr;
    }
}

__device__ void bessel_j_real_miller(double x, int n_max, double* jn) {
    int n_start = n_max + (int)(x) + MILLER_MARGIN;
    if (n_start < n_max + MILLER_MIN_MARGIN) n_start = n_max + MILLER_MIN_MARGIN;
    if (n_start > MILLER_MAX_START) n_start = MILLER_MAX_START;

    double temp[TEMP_ARRAY_SIZE];
    double j_curr = 0.0;
    double j_prev = 1.0;

    temp[n_start] = j_curr;
    temp[n_start - 1] = j_prev;

    for (int n = n_start - 1; n > 0; n--) {
        double j_next = (2.0 * n / x) * j_prev - j_curr;
        j_curr = j_prev;
        j_prev = j_next;
        if (n - 1 <= n_max) {
            temp[n - 1] = j_next;
        }
    }

    double j0_exact = j0(x);
    double scale = j0_exact / temp[0];

    for (int n = 0; n <= n_max; n++) {
        jn[n] = temp[n] * scale;
    }
}

__device__ void bessel_y_real(double x, int n_max, double* yn) {
    yn[0] = y0(x);
    yn[1] = y1(x);

    for (int n = 1; n < n_max; ++n) {
        yn[n + 1] = (2.0 * n / x) * yn[n] - yn[n - 1];
    }
}

__device__ void bessel_j_complex_miller(const complex_t& z, int n_max, complex_t* jn) {
    int n_start = n_max + MILLER_MARGIN;
    if (n_start > MILLER_MAX_START) n_start = MILLER_MAX_START;

    complex_t j_curr = make_complex(0.0, 0.0);
    complex_t j_prev = make_complex(1.0, 0.0);

    complex_t temp[TEMP_ARRAY_SIZE];
    temp[n_start] = j_curr;
    temp[n_start - 1] = j_prev;

    for (int n = n_start - 1; n > 0; n--) {
        complex_t j_next = (2.0 * n / z) * j_prev - j_curr;
        j_curr = j_prev;
        j_prev = j_next;
        if (n - 1 <= n_max) {
            temp[n - 1] = j_next;
        }
    }

    complex_t j0_exact = bessel_j_series(0, z);
    complex_t scale = j0_exact / temp[0];

    for (int n = 0; n <= n_max; n++) {
        jn[n] = temp[n] * scale;
    }
}

#endif
