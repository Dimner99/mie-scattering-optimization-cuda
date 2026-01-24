#ifndef MIE_OPTIMIZER_HELPERS_CUH
#define MIE_OPTIMIZER_HELPERS_CUH

__device__ inline void momentum_update(
    double grad,
    double momentum,
    double lr,
    double* velocity,
    double* out_step)
{
    *velocity = momentum * (*velocity) + (1.0 - momentum) * grad;
    *out_step = lr * (*velocity);
}

#endif
