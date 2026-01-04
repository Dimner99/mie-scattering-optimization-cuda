ClearAll["Global`*"]

$Assumptions = Element[{epsre, epsim, r, lam}, Reals];

(* Define conjc as a placeholder for conjugate *)
SetAttributes[conjc, HoldAll];
conjc /: D[conjc[u_], v_] := conjc[D[u, v]];

(* Define custom derivative rules for Bessel and Hankel functions *)
Unprotect[BesselJ, HankelH2];
BesselJ /: D[BesselJ[nu_, z_], z] := (BesselJ[nu - 1, z] - BesselJ[nu + 1, z]) / 2;
HankelH2 /: D[HankelH2[nu_, z_], z] := (HankelH2[nu - 1, z] - HankelH2[nu + 1, z]) / 2;
Protect[BesselJ, HankelH2];

epsre = Symbol["epsre"];
epsim = Symbol["epsim"];
r = Symbol["r"];
lam = Symbol["lam"];
k0 = 2 Pi/lam;
x = k0*r;
m = Sqrt[epsre + I*epsim];
mx = m*x;

(* ---- Define an coefficients as num/den ---- *)
(* TM mode *)
numTM = m*BesselJ[n, x]*BesselJ[n-1, mx] - BesselJ[n, mx]*BesselJ[n-1, x];
denTM = BesselJ[n, mx]*HankelH2[n-1, x] - m*HankelH2[n, x]*BesselJ[n-1, mx];

(* TE mode *)
termTE0 = (n/x) * (m - 1/m);
numTE = BesselJ[n, x]*BesselJ[n-1, mx] - m*BesselJ[n, mx]*BesselJ[n-1, x] + termTE0*BesselJ[n, x]*BesselJ[n, mx];
denTE = m*BesselJ[n, mx]*HankelH2[n-1, x] - HankelH2[n, x]*BesselJ[n-1, mx] - termTE0*BesselJ[n, mx]*HankelH2[n, x];

(* ---- Compute gradients using quotient rule ---- *)
(* For term = Re(a) + |a|^2 where a = num/den:
   d(term)/dp = Re(da/dp * (1 + 2*conj(a)))
   where da/dp = (d(num)/dp * den - num * d(den)/dp) / den^2 *)

computeGrad[num_, den_, param_] := Module[
  {dnum, dden, dadp},
  dnum = D[num, param];
  dden = D[den, param];
  (* da/dp = (dnum*den - num*dden) / den^2 *)
  dadp = (dnum*den - num*dden) / den^2;
  (* d(term)/dp = Re(dadp * (1 + 2*conj(a))) = Re(dadp * (1 + 2*conj(num/den))) *)
  (* = Re(dadp * (den + 2*conj(num)) / conj(den)) *)
  (* But we output the expression and let CUDA compute the real part *)
  dadp * (1 + 2*conjc[num/den])
];

dqtmdepsre = computeGrad[numTM, denTM, epsre];
dqtmdepsim = computeGrad[numTM, denTM, epsim];
dqtmdr = computeGrad[numTM, denTM, r];

dqtedepsre = computeGrad[numTE, denTE, epsre];
dqtedepsim = computeGrad[numTE, denTE, epsim];
dqtedr = computeGrad[numTE, denTE, r];

termTM = numTM/denTM;
termTE = numTE/denTE;

format[expr_] := StringReplace[
  ToString[expr, InputForm],
  {"Conjugate[" -> "conjc["}
];

out = StringRiffle[
  {
    "term_tm = " <> format[termTM] <> ";",
    "term_te = " <> format[termTE] <> ";",
    "dq_tm_deps_re_n = " <> format[dqtmdepsre] <> ";",
    "dq_tm_deps_im_n = " <> format[dqtmdepsim] <> ";",
    "dq_tm_dr_n = " <> format[dqtmdr] <> ";",
    "dq_te_deps_re_n = " <> format[dqtedepsre] <> ";",
    "dq_te_deps_im_n = " <> format[dqtedepsim] <> ";",
    "dq_te_dr_n = " <> format[dqtedr] <> ";"
  },
  "\n"
];

Export["wolfram_gradients.txt", out, "String"]
