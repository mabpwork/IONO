"""Files of ionolum: spectra (NAME_ionospectra<n>.txt), beam current (NAME_current_K6485_esp_<n>.txt),
spectra without background (NAME_ionospectra<n>_nobg.txt / _norm.txt) and integrated area vs fluence files."""

import os
import re

import numpy as np

from .constants import FLUENCE_UNIT


SPEC_RE = re.compile(r"^(?P<name>.+)_ionospectra(?P<n>\d+)\.txt$", re.I)


CURR_RE = re.compile(r"^(?P<name>.+)_current_K6485_esp_(?P<n>\d+)\.txt$", re.I)


def read_header(line):
    """Line 1 of a spectrum: {"time_ms", "xyz", "date"} (Beam Charge / Current are not valid)."""
    f = [s.strip() for s in line.rstrip("\r\n").split("\t")]
    h = {"time_ms": None, "xyz": None, "date": ""}
    for i, s in enumerate(f):
        try:
            if s.startswith("Total Time"):
                h["time_ms"] = float(f[i + 1])
            elif s.startswith("X:"):
                h["xyz"] = tuple(float(v) for v in f[i + 1:i + 4])
            elif s == "Date":
                h["date"] = f[i + 1]
        except (IndexError, ValueError):
            pass
    return h


def read_spectrum(path):
    """(header, wavelength nm, intensity)."""
    with open(path, encoding="latin-1") as fh:
        h = read_header(fh.readline())
        d = np.loadtxt(fh, usecols=(0, 1), ndmin=2)
    return h, d[:, 0], d[:, 1]


def read_current(path):
    """Readings of the beam current (A) of one spectrum."""
    with open(path, encoding="latin-1") as fh:
        vals = fh.read().split()
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except ValueError:
            pass
    return np.array(out)


def split_name(fname):
    """(NAME, kind, number) of a file of a data group; None if it is not one."""
    for kind, rx in (("spectrum", SPEC_RE), ("current", CURR_RE)):
        m = rx.match(fname)
        if m:
            return m.group("name"), kind, m.group("n")
    return None


def load_group(folder, name):
    """Spectra (sorted by number) of NAME in folder, each with its current file (same rank)."""
    spec, curr = [], []
    for f in os.listdir(folder):
        s = split_name(f)
        if s and s[0] == name:
            (spec if s[1] == "spectrum" else curr).append((int(s[2]), s[2], f))
    spec.sort()
    curr.sort()
    items = []
    for k, (_, ns, f) in enumerate(spec):
        h, x, y = read_spectrum(os.path.join(folder, f))
        it = {"k": k, "n": ns, "file": f, "header": h, "x": x, "y": y, "cfile": None, "current": None}
        if k < len(curr):
            it["cfile"] = curr[k][2]
            it["current"] = read_current(os.path.join(folder, curr[k][2]))
        items.append(it)
    # start time of each spectrum (s): spectra acquired one after the other
    t0 = 0.0
    for it in items:
        it["t0"] = t0
        it["dt"] = (it["header"]["time_ms"] or 0) / 1000.0
        t0 += it["dt"]
        it["charge"] = charge(it)
    return items, len(curr)


def charge(it):
    """Beam charge of one spectrum, Q = ∫ I dt (C): readings evenly spread over the integration
    time, so Q = mean(I) × integration time. None without current readings / time."""
    c = it["current"]
    if c is None or not len(c) or not it["dt"]:
        return None
    return float(np.mean(c)) * it["dt"]


def background_of(it, bg):
    """Background (counts) to subtract from spectrum it: spectrum bg on the wavelengths of it,
    scaled by the ratio of the integration times when they differ."""
    b = bg["y"] if np.array_equal(bg["x"], it["x"]) else np.interp(it["x"], bg["x"], bg["y"])
    if it["dt"] and bg["dt"] and it["dt"] != bg["dt"]:
        b = b * it["dt"] / bg["dt"]
    return b


def nobg_name(it, nobg=True):
    """NAME_ionospectra<n>.txt -> NAME_ionospectra<n>_nobg.txt, or _norm.txt without background
    removed (not read again as spectra)."""
    return os.path.splitext(it["file"])[0] + ("_nobg.txt" if nobg else "_norm.txt")


def write_nobg(path, it, bg, flu=None, q_acc=None):
    """Spectrum it without the background bg: wavelength, counts, background, counts − background
    and, with the beam charge, (counts − background) / Q (counts/nC) and, with flu = (fluence per
    coulomb, beam area mm², charge state), (counts − background) / fluence (counts per 10¹² ions/cm²).
    bg None: background 0 (the normalized columns are the ones of the original spectrum).
    Line 1 also has the start and end time of the spectrum from the start of the first one and,
    with q_acc (charge from the start of the experiment to the end of this spectrum, C), the
    accumulated charge and fluence."""
    b = background_of(it, bg) if bg is not None else np.zeros_like(it["y"])
    d = it["y"] - b
    q = it["charge"]
    head = ["Total Time: %g s" % it["dt"]]
    if q:
        head.append("Beam Charge: %.6g nC" % (q * 1e9))
        if flu:
            head.append("Fluence: %.6g ions/cm2 (beam area at the sample %.6g mm2, charge state %d)"
                        % (q * flu[0], flu[1], flu[2]))
    head += ["Start Time: %g s" % it["t0"], "Elapsed Time: %g s" % (it["t0"] + it["dt"])]
    if q_acc is not None:
        head.append("Accumulated Charge: %.6g nC" % (q_acc * 1e9))
        if flu:
            head.append("Accumulated Fluence: %.6g ions/cm2" % (q_acc * flu[0]))
    head += [it["file"], it["cfile"] or "(no current file)",
             "Background: %s (#%d)" % (bg["file"], bg["k"] + 1) if bg is not None else "Background: none (0)"]
    cols = [it["x"], it["y"], b, d]
    names = ["Wavelength(nm)", "Counts", "background", "no_background"]
    if q:
        cols.append(d / (q * 1e9))
        names.append("no_background_normalized(counts/nC)")
        if flu:
            cols.append(d / (q * flu[0] / FLUENCE_UNIT))
            names.append("no_background_fluence_normalized(counts per 1e12 ions/cm2)")
    with open(path, "w", encoding="latin-1", newline="\n") as fh:
        fh.write("\t".join(head) + "\n")
        fh.write("\t".join(names) + "\n")
        for row in zip(*cols):
            fh.write("\t".join("%.6g" % v for v in row) + "\n")


NOBG_RE = re.compile(r"^(?P<name>.+)_ionospectra(?P<n>\d+)_nobg\.txt$", re.I)


def read_nobg(path):
    """File written by write_nobg: {"header": [(key, value)] of line 1, "info": {key: value},
    "cols": column names, "data": array}. The two file names without key in line 1 are
    "Spectrum file" and "Current file"."""
    with open(path, encoding="latin-1") as fh:
        line1 = fh.readline()
        cols = [c.strip() for c in fh.readline().rstrip("\r\n").split("\t") if c.strip()]
        data = np.loadtxt(fh, ndmin=2)
    head, bare = [], ["Spectrum file", "Current file"]
    for f in line1.rstrip("\r\n").split("\t"):
        f = f.strip()
        if not f:
            continue
        if ":" in f:
            k, v = f.split(":", 1)
            head.append((k.strip(), v.strip()))
        else:
            head.append((bare.pop(0) if bare else "", f))
    return {"header": head, "info": dict(head), "cols": cols[:data.shape[1]], "data": data}


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def info_number(info, key):
    """First number of the value of key in line 1 of a nobg file (e.g. Fluence), or None."""
    m = FLOAT_RE.search(info.get(key, ""))
    return float(m.group()) if m else None


def hms(s):
    s = int(round(s))
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def read_area_file(path):
    """File of t_save: {"path", "fluence", "area", "info": {key: value of the # lines}}."""
    info, rows = {}, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for L in fh:
            if L.startswith("#"):
                if ":" in L:
                    k, v = L[1:].split(":", 1)
                    info[k.strip()] = v.strip()
                continue
            f = L.split("\t")
            try:
                rows.append((float(f[0]), float(f[1])))
            except (ValueError, IndexError):
                pass                    # line of the column names
    a = np.array(rows, float).reshape(-1, 2)
    return {"path": path, "fluence": a[:, 0], "area": a[:, 1], "info": info}
