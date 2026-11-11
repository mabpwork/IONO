"""Models fitted to the integrated area vs fluence curves (Data treatment › Models) and helpers."""

import re

import numpy as np


def trapz(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz        # numpy 2 / 1
    return float(f(y, x)) if len(x) > 1 else 0.0


MODEL_FUNCS = {"exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs,
               "sin": np.sin, "cos": np.cos, "tan": np.tan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
               "arctan": np.arctan, "pi": np.pi}


def eval_model(expr, params, x):
    """Value of the model y(x) typed (parameters a0 … a9 = params, numpy functions of MODEL_FUNCS,
    ^ or ** powers) at x (array). ValueError if it cannot be evaluated."""
    e = expr.strip().replace("^", "**")
    if not e:
        raise ValueError("empty expression")
    for name in re.findall(r"[A-Za-z_]\w*", e):
        if name != "x" and name not in MODEL_FUNCS and not re.fullmatch(r"a\d", name) \
                and not re.fullmatch(r"e\d*", name):
            raise ValueError("unknown name '%s'" % name)
    ns = dict(MODEL_FUNCS, x=np.asarray(x, float), __builtins__={})
    ns.update(("a%d" % k, float(v)) for k, v in enumerate(params))
    try:
        with np.errstate(all="ignore"):
            y = eval(e, ns)
    except Exception as err:
        raise ValueError(str(err))
    return np.broadcast_to(np.asarray(y, float), np.shape(x)).copy()


MODEL_PRESETS = {
    "Custom": "",
    "Birks–Black  a0 / (1 + a1·x)": "a0/(1+a1*x)",
    # P. A. Sullivan, R. A. Baragiola, J. Appl. Phys. 76, 4847 (1994), Eq. (6) + an additive
    # constant (background): L = L0 / {1 + k [exp(σD) − 1]} + C; radiative centres
    # Nr = N0 exp(−σD) are damaged, x = (N0 − Nr)/Nr = exp(σD) − 1; a1 = k (non-radiative /
    # radiative transition rates), a2 = σ (effective damage cross section), a3 = C
    "Sullivan–Baragiola  a0 / (1 + a1·(exp(a2·x) − 1)) + a3": "a0/(1+a1*(exp(a2*x)-1))+a3",
    "Exponential  a0·exp(−a1·x) + a2": "a0*exp(-a1*x)+a2",
    "Two exponentials": "a0*exp(-a1*x)+a2*exp(-a3*x)+a4",
    "Linear  a0 + a1·x": "a0+a1*x",
    "Power  a0·x^a1": "a0*x**a1"}
X_UNITS = {"ions/cm²": 1.0, "10¹² ions/cm²": 1e12, "10¹⁵ ions/cm²": 1e15}
DEPTH_UNITS = {"nm": 10.0, "µm": 1e4}             # Å per unit
