# Mie Cylinder Optimizer

This project solves an inverse Mie-scattering problem for infinite dielectric cylinders. Given a radius $a$ and a wavelength band, it searches for the complex permittivity $\varepsilon = \varepsilon_{re} + i\,\varepsilon_{im}$ that maximizes the spectral-averaged absorption efficiency $Q_{\text{abs}}$.
See the ***main.wlnb*** for the problem statement and solution. All the calculations are being done using gpu and custom cuda c++ kernels or jax gpu cabable code.

We are using two approaches, brute force optimization where we copmute the
objective function on the whole grid of parameters and gradient descent using both 
symbolically generated gradients and leveraging the jax autodiff capabilities.
For the symbolic we use a pipeline to dynamically generate the cuda kernels using sophisticated parser to read the gradients formulas and transform and embed them to cuda template files.

## What Lives Where

**Core physics and kernels**

- ***mie_kernel.cu*** evaluates spectrally-averaged $Q_{\text{abs}}^{TM}$ and $Q_{\text{abs}}^{TE}$ over an $(\varepsilon_{re}, \varepsilon_{im})$ grid for a single radius.
- ***mie_bessel_helpers.cuh*** provides complex arithmetic plus the Bessel recurrences and power-series seeds for $J_0$/$Y_0$/$Y_1$.
See [Bessel_Computations.md](Bessel_Computations.md)
- ***mie_optimizer_helpers.cuh*** contains shared helpers for the gradient kernels.
- ***cpu_mie.py*** is the NumPy/SciPy reference, vectorized across orders, wavelengths, and $\varepsilon$ grid points.

**Brute-force GPU search**




- ***gpu_mie.py*** is a CuPy wrapper that launches the grid-evaluation kernel and extracts the per-radius optimum.

**Gradient-based optimization**

- For the gradient descent we use an SGD optimizer with momentum.

- ***export_gradients.wls*** derives $\partial Q / \partial \varepsilon_{re}$, $\partial Q / \partial \varepsilon_{im}$, and $\partial Q / \partial a$ symbolically in Mathematica, producing ***wolfram_gradients.txt***.
- ***wolfram_gradient_parser.py*** parses those expressions into SymPy, and ***cuda_gradient_codegen.py*** emits CUDA, filling in ***mie_kernel_grad_generated.cu*** and ***mie_kernel_grad_batch_generated.cu*** from the templates in ***mie_kernel_grad_template.cu*** and ***mie_kernel_grad_batch.cu***.
- ***gpu_gradient.py*** wraps both generated kernels, while ***gradient_descent.py*** orchestrates multi-radius optimization.

**Main files**

- ***main.py*** runs brute force and momentum-SGD side-by-side, then plots the comparison.
- ***jax_version.py*** mirrors the full pipeline in JAX with `lax.scan`, automatic differentiation, and `vmap` to produce a highly optimized version in jax. The autodiff works perfect for the bessel functions as we used the series expansions and the recurrence realtions to compute them,See [Bessel_Computations.md](Bessel_Computations.md)



