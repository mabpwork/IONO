"""Beam at the sample for an external beam: gas density, TRIM run through the window and the gas
(TRANSMIT.txt), energy and area of the beam at the sample, fluence."""

import os
import subprocess

import numpy as np

from .constants import ATMOSPHERES, E_CHARGE, R_GAS, SPOT_FRACTION, SRIM_DIR
from .data_files import FLOAT_RE


def fluence_per_coulomb(area_mm2, charge_state=1):
    """Fluence (ions/cm²) per coulomb of beam charge: ions = Q / (q·e) on the area of the beam."""
    return 1.0 / (charge_state * E_CHARGE * area_mm2 * 1e-2)


def gas_density(atm, p_mbar=1013.25, t_c=20.0):
    """Density (g/cm³) of the atmosphere atm (ATMOSPHERES) at p (mbar) and T (°C), ideal gas."""
    return p_mbar * 100.0 * ATMOSPHERES[atm][1] / (R_GAS * (t_c + 273.15)) / 1e6     # g/m³ -> g/cm³


def trim_run_dir(folder=SRIM_DIR):
    """SRIM folder where TRIM runs (TRIM.exe, TRIM.IN and the outputs); it must be writable."""
    if not os.path.exists(os.path.join(folder, "TRIM.exe")):
        raise FileNotFoundError("TRIM.exe not found in %s" % folder)
    if not os.access(folder, os.W_OK):
        raise PermissionError("the SRIM folder cannot be written: %s" % folder)
    return folder


def read_transmit(path):
    """Transmitted ions of TRANSMIT.txt: array of rows (energy eV, X, Y, Z (Å), cos X, cos Y, cos Z)."""
    rows = []
    with open(path, encoding="latin-1") as fh:
        for L in fh:
            if L.startswith("T"):
                v = FLOAT_RE.findall(L[1:])
                if len(v) >= 9:
                    rows.append([float(s) for s in v[2:9]])
    return np.array(rows).reshape(-1, 7)


def run_trim_transmit(ion, mass, e_kev, layers, n_ions, run_dir, proc=None):
    """TRIM of the ion (symbol, mass u, energy keV) through layers [(name, {el: fraction}, ρ g/cm³,
    width Å, gas)], in run_dir (no window, minimized); returns read_transmit of the ions that cross
    them. proc: list where the Popen is put (to stop it)."""
    from .srim_tools import import_srim, trim_in_text
    srim = import_srim()
    lays = [srim.Layer({s: {"stoich": f, "E_d": 25.0, "lattice": 3.0, "surface": 2.0} for s, f in els.items()},
                       density=rho, width=w, phase=1 if gas else 0, name=name)
            for name, els, rho, w, gas in layers]
    trim = srim.TRIM(srim.Target(lays), srim.Ion(ion, energy=e_kev * 1e3, mass=mass), calculation=1,
                     number_ions=int(n_ions), plot_mode=5, transmit=1, description="ionolum beam at the sample")
    out = os.path.join(run_dir, "SRIM Outputs", "TRANSMIT.txt")
    if os.path.exists(out):
        os.remove(out)
    with open(os.path.join(run_dir, "TRIM.IN"), "w", newline="\r\n") as fh:
        fh.write(trim_in_text(trim))
    with open(os.path.join(run_dir, "TRIMAUTO"), "w", newline="\r\n") as fh:
        fh.write("1\n")
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 4                                  # SW_SHOWNOACTIVATE: shown, does not take the focus
    # TRIM.exe of Program Files is marked "run as administrator" (compatibility flag): RunAsInvoker
    # runs it without the UAC prompt (the SRIM folder is writable by the users)
    env = dict(os.environ, __COMPAT_LAYER="RunAsInvoker")
    p = subprocess.Popen([os.path.join(run_dir, "TRIM.exe")], cwd=run_dir, startupinfo=si, env=env)
    if proc is not None:
        proc.append(p)
    p.wait()
    if p.returncode:
        raise RuntimeError("TRIM stopped (code %s)" % p.returncode)
    if not os.path.exists(out):
        raise RuntimeError("TRIM did not write TRANSMIT.txt")
    return read_transmit(out)


def beam_spot(d0_mm, yz=None, n=20000, seed=1):
    """Positions (mm) of n ions at the sample: uniform disc of diameter d0 at the nozzle plus the
    lateral deviations yz (mm) of TRIM ions taken at random."""
    rng = np.random.default_rng(seed)
    r0 = d0_mm / 2
    ph, rr = rng.uniform(0, 2 * np.pi, n), r0 * np.sqrt(rng.uniform(0, 1, n))
    pts = np.c_[rr * np.cos(ph), rr * np.sin(ph)]
    if yz is not None and len(yz):
        pts = pts + yz[rng.integers(0, len(yz), n)]
    return pts


def beam_at_sample(tr, n_ions, d0_mm):
    """Energy and area of the beam at the sample from the transmitted ions tr (read_transmit) of a
    TRIM run of n_ions point-like ions, for a beam of uniform circular section of diameter d0 (mm)
    at the nozzle. Diameter at the sample: the beam spot (beam_spot) is compared with the disc
    at the nozzle by the radius of the circle with SPOT_FRACTION of the ions,
    d = d0 · r95(sample) / r95(nozzle), r95(nozzle) = R0·√0.95 (no deviation: d = d0); A = π d²/4."""
    if not len(tr):
        raise RuntimeError("no ion reached the sample")
    e = tr[:, 0] / 1e3
    yz = tr[:, 2:4] * 1e-7                                 # Å -> mm
    pts = beam_spot(d0_mm, yz)
    rf = float(np.percentile(np.hypot(pts[:, 0], pts[:, 1]), 100 * SPOT_FRACTION))
    d = 2 * rf / np.sqrt(SPOT_FRACTION)
    a0 = np.pi * d0_mm ** 2 / 4
    r2 = (yz ** 2).sum(axis=1)
    ang = np.degrees(np.arccos(np.clip(tr[:, 4], -1, 1)))
    return {"E": float(np.mean(e)), "E_sd": float(np.std(e)), "A0": a0, "A": float(np.pi * d * d / 4),
            "d": float(d), "sigma": float(np.sqrt(np.mean(r2) / 2)), "angle": float(np.sqrt(np.mean(ang ** 2))),
            "frac": len(tr) / float(n_ions), "e": e, "yz": yz, "pts": pts}
