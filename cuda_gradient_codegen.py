from __future__ import annotations

from pathlib import Path
from sympy import Symbol, I
from sympy.printing.c import C99CodePrinter
from wolfram_gradient_parser import parse_wolfram_gradients_file


# Wolfram symbol -> CUDA variable name mapping
SYMBOL_MAP = {
    "lam": "wav",
    "r": "a",
    "epsre": "eps_re",
    "epsim": "eps_im",
}


class CudaPrinter(C99CodePrinter):
    
    def _print_Symbol(self, expr):
        return SYMBOL_MAP.get(expr.name, expr.name)

    def _print_ImaginaryUnit(self, expr):
        return "make_complex(0.0, 1.0)"

    def _print_Pi(self, expr):
        return "M_PI"

    def _print_Pow(self, expr):
        from sympy import Add
        base, exp = expr.as_base_exp()
        base_str = self._print(base)
        if isinstance(base, Add):
            base_str = f"({base_str})"
        
        # Small positive integer powers, do not use the pow as it is slower
        if exp.is_integer and exp.is_positive and exp <= 6:
            expanded = " * ".join([base_str] * int(exp))
            return f"({expanded})" if int(exp) > 1 else expanded
        
        # Small negative integer powers -> 1 / (base * base * ...)
        if exp.is_integer and exp.is_negative and exp >= -6:
            denom = " * ".join([base_str] * int(-exp))
            return f"1.0 / ({denom})"
        
        # exp == 0
        if exp == 0:
            return "1.0"
        
        # Complex base needs cpow
        if base.has(I) or (hasattr(base, 'is_complex') and base.is_complex):
            return f"cpow({self._print(base)}, {self._print(exp)})"
        return f"pow({self._print(base)}, {self._print(exp)})"

    def _print_Function(self, expr):
        name = expr.func.__name__
        
        if name == "BesselJ":
            order, arg = expr.args
            # Detect if argument contains eps (-> complex mx argument)
            epsim_sym = Symbol('epsim')
            epsre_sym = Symbol('epsre')
            is_complex_arg = epsim_sym in arg.free_symbols or epsre_sym in arg.free_symbols
            arr = "jn_mx" if is_complex_arg else "jn_x"
            order_str = self._print(order)
            return f"{arr}_at((int)({order_str}), {arr})"
        
        if name == "HankelH2":
            order, _ = expr.args
            order_str = self._print(order)
            return f"h2n_x_at((int)({order_str}), h2n_x)"
        
        if name == "anTM":
            return "an_tm"
        
        if name == "anTE":
            return "an_te"
        
        if name == "conjc":
            return f"conjc({self._print(expr.args[0])})"
        
        return super()._print_Function(expr)


def generate_cuda_gradient_kernel(
    gradients_path: Path,
    template_path: Path,
    output_path: Path,
) -> None:

    gradients = parse_wolfram_gradients_file(str(gradients_path))
    printer = CudaPrinter()

    def gen_line(expr, var_name):
        if expr is None or expr == 0:
            return f"{var_name} = 0.0;"
        # Here we place the missing real from the custom derivative rules in the 
        # wolfram export_gradients.wls file
        return f"{var_name} = real({printer.doprint(expr)});" 

    tm_block = "\n".join([
        gen_line(gradients.dq_tm_deps_re_n, "dq_tm_deps_re_n"),
        gen_line(gradients.dq_tm_deps_im_n, "dq_tm_deps_im_n"),
        gen_line(gradients.dq_tm_dr_n, "dq_tm_dr_n"),
    ])
    
    te_block = "\n".join([
        gen_line(gradients.dq_te_deps_re_n, "dq_te_deps_re_n"),
        gen_line(gradients.dq_te_deps_im_n, "dq_te_deps_im_n"),
        gen_line(gradients.dq_te_dr_n, "dq_te_dr_n"),
    ])

    template = template_path.read_text(encoding="utf-8")
    rendered = (
        template.replace("// @GRAD_TM_EXPR", tm_block)
        .replace("// @GRAD_TE_EXPR", te_block)
    )

    output_path.write_text(rendered, encoding="utf-8")
    print(f"Generated: {output_path}")


def generate_batched_kernel(
    gradients_path: Path,
    template_path: Path,
    output_path: Path,
) -> None:
    """Generate batched gradient descent CUDA kernel.
    
    This kernel runs the full gradient descent loop on GPU,
    processing all radii and restarts in parallel.
    """
    gradients = parse_wolfram_gradients_file(str(gradients_path))
    printer = CudaPrinter()

    def gen_line(expr, var_name):
        if expr is None or expr == 0:
            return f"{var_name} = 0.0;"
        return f"{var_name} = real({printer.doprint(expr)});"

    # Only need eps gradients for optimization (not radius)
    tm_eps_block = "\n                ".join([
        gen_line(gradients.dq_tm_deps_re_n, "dq_deps_re_n"),
        gen_line(gradients.dq_tm_deps_im_n, "dq_deps_im_n"),
    ])
    
    te_eps_block = "\n                ".join([
        gen_line(gradients.dq_te_deps_re_n, "dq_deps_re_n"),
        gen_line(gradients.dq_te_deps_im_n, "dq_deps_im_n"),
    ])

    template = template_path.read_text(encoding="utf-8")
    rendered = (
        template.replace("// @GRAD_TM_EPS_EXPR", tm_eps_block)
        .replace("// @GRAD_TE_EPS_EXPR", te_eps_block)
    )

    output_path.write_text(rendered, encoding="utf-8")
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CUDA gradient kernel from Wolfram output")
    parser.add_argument("--input", required=True, help="Path to Wolfram gradient text file")
    parser.add_argument("--template", default="mie_kernel_grad_template.cu")
    parser.add_argument("--output", default="mie_kernel_grad_generated.cu")
    parser.add_argument("--batch-template", default="mie_kernel_grad_batch.cu")
    parser.add_argument("--batch-output", default="mie_kernel_grad_batch_generated.cu")
    args = parser.parse_args()

    generate_cuda_gradient_kernel(
        Path(args.input),
        Path(args.template),
        Path(args.output),
    )
    
    generate_batched_kernel(
        Path(args.input),
        Path(args.batch_template),
        Path(args.batch_output),
    )
