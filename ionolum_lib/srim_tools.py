"""SRIM / TRIM tools (pysrim, TRIM.IN, SRIM output files, SR Module, compounds, isotopes, cursor).

Copied from IBIST.py (the functions used by the SRIM tab of ionolum and the ones they call) so that
ionolum does not depend on IBIST.py; they can now be changed here independently.
"""

import os
import re

import numpy as np


def isotope_masses(symbol):
    """Isotopes of an element (periodictable package) as a list of
    (mass number, mass in u, abundance %): the natural ones, most abundant
    first (if none is natural: all known isotopes, the one closest to the
    standard atomic mass first). [] if the symbol is not an element.
    Accepts any case ('he', 'HE'); 'n' is the neutron."""
    s = symbol.strip()
    if s != "n":
        s = s[:1].upper() + s[1:].lower()
    if not s:
        return []
    try:
        import periodictable
        el = periodictable.elements.symbol(s)
    except (ImportError, ValueError):
        return []
    isos = [el[a] for a in el.isotopes]
    natural = [i for i in isos if i.abundance > 0]
    if natural:
        isos = sorted(natural, key=lambda i: -i.abundance)
    else:
        isos = sorted(isos, key=lambda i: abs(i.mass - el.mass))
    return [(i.isotope, i.mass, i.abundance) for i in isos]


_MOL_TOKEN = re.compile(r"([A-Z][a-z]?)(\?=)?(\d+(?:\.\d+)?|\.\d+)?")


def parse_molecule(formula):
    """Formula such as 'H2O', 'H2O?=1' or 'H?=2O' -> list of
    (symbol, number of atoms, variable). 'X?=n' means: the amount of X can
    vary in the NDF fit, up to n atoms; a plain number is a fixed amount
    (1 if missing). Raises ValueError for anything else or an unknown
    element."""
    f = formula.replace(" ", "")
    if not f:
        raise ValueError("empty formula")
    out, pos = [], 0
    while pos < len(f):
        m = _MOL_TOKEN.match(f, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot read '{f[pos:]}'")
        sym, var, num = m.group(1), bool(m.group(2)), m.group(3)
        if not isotope_masses(sym):
            raise ValueError(f"unknown element '{sym}'")
        if var and not num:
            raise ValueError(f"'{sym}?=' needs a maximum")
        out.append((sym, float(num) if num else 1.0, var))
        pos = m.end()
    return out


# colours of the layers in the drawing of the target (NDF > STR)
LAYER_COLORS = ["#4472c4", "#ffc000", "#ff0000", "#70ad47", "#7030a0",
                "#00b0f0", "#ed7d31", "#a5a5a5"]


# fixed categorical palette (each file keeps its colour)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


N_AVOGADRO = 6.02214076e23


# TRIM calculation types (TRIM.IN 'Cascades' option)
TRIM_CALCS = ["1 - Ion distribution and quick damage (Kinchin-Pease)",
              "2 - Detailed calculation with full damage cascades",
              "3 - Monolayer collision steps / surface sputtering",
              "4 - Ions with given energy/angle/depth, quick (TRIM.DAT)",
              "5 - Ions with given energy/angle/depth, full (TRIM.DAT)",
              "6 - Recoil cascades from neutrons etc., quick",
              "7 - Recoil cascades and monolayer steps, full"]


TRIM_PLOTS = ["0 - Ion distribution, recoils projected on y-plane",
              "1 - Ion distribution, recoils projected on z-plane",
              "2 - Ion distribution without recoils, y-plane",
              "3 - Transverse plot (yz-plane)",
              "4 - All four on one screen",
              "5 - No graphics (faster)"]


TRIM_COLLISIONS = ["0 - No", "1 - Ions", "2 - Ions and recoils"]


# SRIM output files: (description, kind, names of the columns after the
# depth, name of the columns per target atom that follow them).
# kind: 'depth' table vs depth, 'range3d' final ion positions, 'energy'
# list of particles (histogram of their energy), 'text' not plotted.
SRIM_OUTPUTS = {
    "IONIZ.txt": ("Ionization: energy loss to electrons (eV/(Å·ion))",
                  "depth", ["Ions", "Recoils"], None),
    "PHONON.txt": ("Phonons (phonons/(Å·ion))", "depth",
                   ["Ions", "Recoils"], None),
    "E2RECOIL.txt": ("Energy to recoils (eV/(Å·ion))", "depth",
                     ["Ions"], "Absorbed"),
    "VACANCY.txt": ("Vacancies (vacancies/(Å·ion))", "depth",
                    ["Knock-ons"], "Vacancies"),
    "NOVAC.txt": ("Replacement collisions (/(Å·ion))", "depth",
                  ["Replacements"], None),
    "RANGE.txt": ("Final distribution ((atoms/cm³)/(atoms/cm²))", "depth",
                  ["Ions"], "Recoils"),
    "LATERAL.txt": ("Lateral spread of the ions (Å)", "depth",
                    ["Lateral proj. range", "Projected straggling",
                     "Lateral radial"], None),
    "RANGE_3D.txt": ("Final positions of the ions (depth histogram)",
                     "range3d", None, None),
    "BACKSCAT.txt": ("Backscattered ions (energy histogram)", "energy",
                     None, None),
    "TRANSMIT.txt": ("Transmitted ions (energy histogram)", "energy",
                     None, None),
    "SPUTTER.txt": ("Sputtered atoms (energy histogram)", "energy",
                    None, None),
    "COLLISON.txt": ("Collision details", "text", None, None),
    "TRIMOUT.txt": ("TRIM summary", "text", None, None),
    "TDATA.txt": ("TRIM data", "text", None, None),
    "TRIM.IN": ("TRIM input file", "text", None, None),
}


def import_srim():
    """Import pysrim. Its element database is read with yaml.load(stream)
    without a Loader, which recent PyYAML refuses: a Loader is supplied
    while srim is imported. Raises ImportError if pysrim is missing."""
    import yaml
    orig = yaml.load

    def load(stream, Loader=None, **kw):
        return orig(stream, Loader=Loader or yaml.FullLoader, **kw)
    yaml.load = load
    try:
        import srim
        import srim.input
        import srim.output
    finally:
        yaml.load = orig
    return srim


def trim_in_text(trim):
    """Text of the TRIM.IN file of a pysrim TRIM object (the same as
    TRIMInput.write, without writing in the current folder), with '\\n'
    line ends: pysrim writes '\\r\\n', which a file opened with
    newline='\\r\\n' would turn into '\\r\\r\\n' (TRIM then reads empty lines
    and stops with 'run-time error 62: input past end of file')."""
    srim = import_srim()
    ti = srim.input.TRIMInput(trim)
    return trim_in_lf("".join(m() for m in (
        ti._write_title, ti._write_ion, ti._write_cascade_options,
        ti._write_plot_on_off, ti._write_target, ti._write_plot_options,
        ti._write_elements, ti._write_layer, ti._write_solid_gas,
        ti._write_bragg_correction, ti._write_displacement_energies,
        ti._write_lattice_binding, ti._write_surface_binding,
        ti._write_version)))


def trim_in_lf(text):
    """'\\n' line ends only (the files are written with newline='\\r\\n'),
    and a final line end."""
    t_ = text.replace("\r\n", "\n").replace("\r", "\n")
    return t_ if t_.endswith("\n") else t_ + "\n"


def layer_density(elements, molecules=(), rho_typed=None):
    """(atomic density in 1e22 atoms/cm3, mass density in g/cm3, mean
    atomic mass) of a layer of elements [(symbol, fraction, ...)].
    The atomic density is the one of the molecules of the layer (NDF
    density in 1e22 at/cm3; mean weighted by the molecule factors) or,
    without them, the density rho_typed (g/cm3) given for the layer, or
    the one of the pure elements (periodictable densities) mixed by
    volume. None for what is unknown."""
    import periodictable
    els = [(e[0], float(e[1])) for e in elements if e[0] and e[1] > 0]
    tot = sum(c for _, c in els)
    if not els or tot <= 0:
        return None, None, None
    try:
        pe = [(periodictable.elements.symbol(element_symbol(s)), c / tot)
              for s, c in els]
    except ValueError:
        return None, None, None
    m_avg = sum(el.mass * c for el, c in pe)
    n = None
    dens = []
    for m in molecules:
        try:
            f = str(m[2]).replace(" ", "") if len(m) > 2 else "1"
            f = f[2:] if f.startswith("?=") else f      # fitted factor
            d, f = float(str(m[1]).replace(",", ".")), float(f or 1)
        except (ValueError, TypeError, IndexError):
            continue
        if d > 0 and f > 0:
            dens.append((d, f))
    if dens:
        n = sum(d * f for d, f in dens) / sum(f for _, f in dens)
    elif rho_typed:                        # density given for the layer
        n = rho_typed * N_AVOGADRO / m_avg / 1e22
    else:
        inv = 0.0                     # 1/rho = sum(w_i / rho_i)
        for el, c in pe:
            if not el.density:
                inv = None
                break
            inv += el.mass * c / m_avg / el.density
        if inv:
            n = 1.0 / inv * N_AVOGADRO / m_avg / 1e22
    rho = n * 1e22 * m_avg / N_AVOGADRO if n else None
    return n, rho, m_avg


def element_symbol(sym):
    """'29Si' -> 'Si', 'D'/'T' -> 'H', 'si' -> 'Si' (None if empty)."""
    s_ = re.sub(r"^\d+", "", str(sym).strip())
    s_ = s_[:1].upper() + s_[1:].lower()
    return ("H" if s_ in ("D", "T") else s_) or None


def srim_table(path, max_rows=None):
    """Numeric table of a SRIM output file: the rows after the line of
    dashes, as a numpy array (rows x columns). Read with pysrim when it
    can (it reads only the 100 depth bins), else here (also with decimal
    commas)."""
    with open(path, "rb") as f:
        raw = f.read()
    if max_rows is None:
        try:
            data = import_srim().output.SRIM_Output()._read_table(raw)
            if data.ndim == 2 and len(data):
                return data
        except Exception:
            pass
    rows, started = [], False
    for ln in raw.decode("latin-1").splitlines():
        if re.match(r"\s*-{3,}(\s+-{3,})+\s*$", ln):
            started, rows = True, []
            continue
        if not started:
            continue
        toks = ln.split()
        try:
            vals = [float(t.replace(",", ".")) for t in toks]
        except ValueError:
            vals = []
        if len(vals) < 2:
            if rows:
                break
            continue
        rows.append(vals)
        if max_rows and len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError("no table found")
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows])


def srim_energies(path):
    """Energies (keV) of the particles in BACKSCAT/TRANSMIT/SPUTTER.txt
    (lines 'B', 'T' or 'S', particle number, atom, energy in eV, ...)."""
    out = []
    with open(path, "r", encoding="latin-1") as f:
        for ln in f:
            m = re.match(r"\s*[BTS]\s*(\d.*)", ln)
            if not m:
                continue
            toks = m.group(1).split()
            try:
                out.append(float(toks[2].replace(",", ".")) / 1e3)
            except (ValueError, IndexError):
                pass
    return np.array(out)


def read_trim_in(path):
    """Target atoms ['Si', 'O', ...] and layer widths (Å) of a TRIM.IN
    file ([], [] if it cannot be read)."""
    atoms, widths = [], []
    try:
        with open(path, "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
    except OSError:
        return atoms, widths
    for ln in lines:
        m = re.match(r"\s*Atom\s+\d+\s*=\s*([A-Za-z]+)", ln)
        if m:
            atoms.append(m.group(1))
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("numb"):
            for l2 in lines[i + 1:]:
                m = re.match(r'\s*\d+\s+"[^"]*"\s+([-+\d.eE]+)', l2)
                if not m:
                    break
                widths.append(float(m.group(1)))
            break
    return atoms, widths


def read_trim_in_layers(path):
    """Layers of a TRIM.IN file: [{name, width (Å), density (g/cm3),
    comp {symbol: stoichiometry}, gas}] in depth order. The stoichiometry
    columns follow the 'Atom i = X' lines. [] if it cannot be read."""
    try:
        with open(path, "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    atoms, masses = [], []
    for ln in lines:
        m = re.match(r"\s*Atom\s+\d+\s*=\s*([A-Za-z]+)\s*=\s*\d+\s+"
                     r"([-+\d.eE]+)", ln) or \
            re.match(r"\s*Atom\s+\d+\s*=\s*([A-Za-z]+)", ln)
        if m:
            atoms.append(m.group(1))
            try:
                masses.append(float(m.group(2)) if m.lastindex > 1
                              else None)
            except ValueError:
                masses.append(None)
    layers = []
    for i, ln in enumerate(lines):
        if not ln.strip().lower().startswith("numb"):
            continue
        for l2 in lines[i + 1:]:
            m = re.match(r'\s*\d+\s+"([^"]*)"\s+(.*)$', l2)
            if not m:
                break
            try:
                vals = [float(t) for t in m.group(2).split()]
            except ValueError:
                break
            if len(vals) < 2:
                break
            comp = {}
            m_sum = c_sum = 0.0            # mean atomic mass of the layer
            for s_, ma, c in zip(atoms, masses, vals[2:]):
                if c > 0:
                    comp[s_] = comp.get(s_, 0.0) + c
                    if ma:
                        m_sum += ma * c
                        c_sum += c
            # atomic density (atoms/cm3) = rho·N_A / mean mass
            n_at = (vals[1] * N_AVOGADRO / (m_sum / c_sum)
                    if c_sum > 0 and m_sum > 0 else None)
            layers.append({"name": m.group(1).strip(), "width": vals[0],
                           "density": vals[1], "comp": comp, "gas": False,
                           "n": n_at})
        break
    # phases: the line after '... Target layer phases (0=Solid, 1=Gas)'
    for i, ln in enumerate(lines[:-1]):
        if "phases" in ln.lower():
            ph = lines[i + 1].split()
            for L, p in zip(layers, ph):
                L["gas"] = p == "1"
            break
    return layers


def srim_parse_comp(text):
    """Composition typed in SRIM > Structure: 'Si 1 O 2', 'Si 0.33 O 0.67'
    (symbol, amount pairs) or a formula 'SiO2'; also the form shown in the
    list, 'SiO2 (Si 0.333 O 0.667)'. Returns {symbol: fraction}; '29Si'
    counts as Si and D/T as H. ValueError if it cannot be read."""
    t_ = text.strip()
    m = re.match(r".*\((.*)\)\s*$", t_)
    if m:
        t_ = m.group(1)
    toks = t_.replace(",", " ").replace(";", " ").split()
    pairs = []
    if toks and len(toks) % 2 == 0:
        try:
            pairs = [(toks[k], float(toks[k + 1]))
                     for k in range(0, len(toks), 2)]
        except ValueError:
            pairs = []
    if not pairs:
        try:                           # D2O: D/T read as H
            mol = parse_molecule(re.sub(r"[DT](?![a-z])", "H", t_))
        except Exception as e:
            raise ValueError(str(e))
        pairs = [(sym, float(n)) for sym, n, _ in mol]
    out = {}
    for sym, c in pairs:
        sym = re.sub(r"^\d+", "", sym)
        sym = sym[:1].upper() + sym[1:].lower()
        sym = "H" if sym in ("D", "T") else sym
        if not isotope_masses(sym):
            raise ValueError(f"'{sym}' is not an element.")
        if c < 0:
            raise ValueError("negative amount.")
        out[sym] = out.get(sym, 0.0) + c
    tot = sum(out.values())
    if tot <= 0:
        raise ValueError("no element.")
    return {k: v / tot for k, v in out.items()}


def read_atomdata(path):
    """SRIM's Data/ATOMDATA: {symbol: {'Z', 'name', 'mass', 'density',
    'heat_subl'}} - 'Heat Subl' (eV) is the surface binding energy that
    SRIM proposes for the element."""
    out = {}
    with open(path, encoding="latin-1") as fh:
        for ln in fh:
            t_ = ln.split()
            if len(t_) < 12 or not t_[0].isdigit():
                continue
            try:
                out[t_[1]] = dict(Z=int(t_[0]), name=t_[2],
                                  mass=float(t_[5]), density=float(t_[6]),
                                  heat_subl=float(t_[-3]))
            except ValueError:
                continue
    return out


def trim_in_atom_energies(path):
    """{symbol: [E_d, lattice, surface]} (eV) of a TRIM.IN: the atoms
    'Atom i = X = Z mass' and the lines after 'displacement energies',
    'lattice binding energies' and 'surface binding energies' (same order;
    the first value of an element repeated in several layers is kept)."""
    with open(path, encoding="latin-1") as fh:
        lines = fh.read().splitlines()
    atoms = []
    for ln in lines:
        m = re.match(r"\s*Atom\s+\d+\s*=\s*([A-Za-z]{1,2})\s*=", ln)
        if m:
            atoms.append(m.group(1))
    vals = {}
    for key, tag in (("ed", "displacement"), ("lat", "lattice binding"),
                     ("surf", "surface binding")):
        for i, ln in enumerate(lines):
            if tag in ln.lower() and "energ" in ln.lower() and \
                    i + 1 < len(lines):
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                  lines[i + 1])
                vals[key] = [float(x) for x in nums[:len(atoms)]]
                break
    if not atoms or len(vals) < 3 or \
            any(len(v) < len(atoms) for v in vals.values()):
        raise ValueError("no target atoms / energies found")
    out = {}
    for k, s_ in enumerate(atoms):
        out.setdefault(s_, [vals["ed"][k], vals["lat"][k], vals["surf"][k]])
    return out


ATOM_ENERGY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "srim_atom_energies.txt")


def read_atom_energy_table(path=None):
    """User table: lines 'symbol E_d lattice surface' (eV), '#' comments.
    {} if there is no file."""
    path = path or ATOM_ENERGY_FILE
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            t_ = ln.split("#")[0].split()
            if len(t_) >= 4:
                try:
                    out[t_[0]] = [float(x) for x in t_[1:4]]
                except ValueError:
                    pass
    return out


def write_atom_energy_table(table, path=None):
    with open(path or ATOM_ENERGY_FILE, "w", encoding="utf-8") as fh:
        fh.write("# ionolum: SRIM target atom energies (eV) used by default\n"
                 "# element  displacement  lattice_binding  "
                 "surface_binding\n")
        for s_ in sorted(table):
            e_ = table[s_]
            fh.write("%-3s %8g %8g %8g\n" % (s_, e_[0], e_[1], e_[2]))


def read_srim_compounds(path):
    """Compounds of SRIM's Data/Compound.dat. Lines: '!' table headers
    (categories), '*' description, then the data line
    "Name", density (g/cm3), number of elements, Z1, x1, Z2, x2, ...
    (atom %, or mass % when the name starts with '%'), the bond line and
    '$' comments. Returns [dict(category, description, name, density,
    mass_percent, given [(Z, symbol, value)], comp {symbol: atom
    fraction}, gas, comments)]."""
    import csv
    import periodictable
    with open(path, "rb") as fh:
        lines = fh.read().decode("latin-1").splitlines()
    out, cat, desc, i = [], "", "", 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("!"):
            # table header: '!' + box-drawing characters + title + box
            # characters, e.g. '!ÄÄÄ  METAL  ALLOYS   ÄÄÄÄ',
            # '!ÄÄ  NUMBERED COMPOUNDS (99-277) from ICRU Report 49, ...'
            m = re.match(r"^!([^\sA-Za-z0-9!]+)\s+(.*?)[\s\W]*$", ln)
            if m and re.search(r"[A-Za-z]{3}", m.group(2)):
                t_ = re.sub(r"[^\x20-\x7e]", " ", m.group(2))
                t_ = " ".join(t_.split(" from ")[0].split())
                cat = t_.title() if t_.isupper() else t_
        elif ln.startswith("*"):
            desc = re.sub(r"[^\x20-\x7e]", "", ln[1:]).strip()
        elif ln.startswith('"') and desc:
            try:
                row = next(csv.reader([ln], skipinitialspace=True))
                name = row[0].strip()
                mass = name.startswith("%")
                rho = float(row[1])
                n = int(float(row[2]))
                vals = [float(v) for v in row[3:3 + 2 * n]]
                given = []
                for k in range(n):
                    z = int(round(vals[2 * k]))
                    given.append((z, periodictable.elements[z].symbol,
                                  vals[2 * k + 1]))
            except (ValueError, IndexError, StopIteration, KeyError):
                desc = ""
                i += 1
                continue
            frac = {}
            for z, s_, v in given:
                a = v / periodictable.elements[z].mass if mass else v
                frac[s_] = frac.get(s_, 0.0) + a
            tot = sum(frac.values()) or 1.0
            comments = []
            j = i + 1
            if j < len(lines) and not lines[j].startswith(("$", "*", "!",
                                                            '"')):
                j += 1                          # bond-structure line
            while j < len(lines) and lines[j].startswith("$"):
                c_ = re.sub(r"[^\x20-\x7e]", " ", lines[j][1:]).rstrip()
                if c_.strip():
                    comments.append(c_)
                j += 1
            out.append(dict(category=cat, description=desc,
                            name=name.lstrip("%").strip(), density=rho,
                            mass_percent=mass, given=given,
                            comp={s_: a / tot for s_, a in frac.items()},
                            gas="(gas)" in desc.lower() or rho < 0.01,
                            comments=comments))
            desc = ""
            i = j
            continue
        i += 1
    return out


def run_elevated(exe, params="", cwd=None, show=1, on_start=None):
    """Run a program with administrator rights (Windows asks for them, UAC)
    and wait for it to end; returns its exit code. Needed for TRIM.exe when
    Windows says 'the requested operation requires elevation' (WinError
    740) or when the SRIM folder (Program Files) cannot be written.
    on_start(handle) is called with the process handle once it runs (to be
    able to stop it with TerminateProcess)."""
    import ctypes
    from ctypes import wintypes

    class SHELLEXECUTEINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("fMask", ctypes.c_ulong),
                    ("hwnd", wintypes.HWND), ("lpVerb", wintypes.LPCWSTR),
                    ("lpFile", wintypes.LPCWSTR),
                    ("lpParameters", wintypes.LPCWSTR),
                    ("lpDirectory", wintypes.LPCWSTR),
                    ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE),
                    ("lpIDList", ctypes.c_void_p),
                    ("lpClass", wintypes.LPCWSTR),
                    ("hkeyClass", wintypes.HKEY),
                    ("dwHotKey", wintypes.DWORD),
                    ("hIconOrMonitor", wintypes.HANDLE),
                    ("hProcess", wintypes.HANDLE)]
    info = SHELLEXECUTEINFO()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = 0x40                      # SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = exe
    info.lpParameters = params
    info.lpDirectory = cwd
    info.nShow = show
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFO)]
    if not shell32.ShellExecuteExW(ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())  # 1223: cancelled
    if not info.hProcess:
        return 0
    if on_start:
        on_start(info.hProcess)
    kernel32.WaitForSingleObject(info.hProcess, 0xFFFFFFFF)
    code = wintypes.DWORD()
    kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
    kernel32.CloseHandle(info.hProcess)
    return code.value


def srim_output_folder(srim_dir):
    """Folder where TRIM leaves its output files: 'SRIM Outputs' of the
    SRIM folder (SRIM 2008/2013) if it exists, else the SRIM folder."""
    sub = os.path.join(srim_dir, "SRIM Outputs")
    return sub if os.path.isdir(sub) else srim_dir


def srim_output_files(srim_dir):
    """TRIM output files: all the files of 'SRIM Outputs' and the known
    output files (SRIM_OUTPUTS, not TRIM.IN) of the SRIM folder."""
    out = []
    sub = os.path.join(srim_dir, "SRIM Outputs")
    if os.path.isdir(sub):
        out += [os.path.join(sub, f) for f in os.listdir(sub)
                if os.path.isfile(os.path.join(sub, f))]
    for name in SRIM_OUTPUTS:
        p_ = os.path.join(srim_dir, name)
        if name != "TRIM.IN" and os.path.isfile(p_):
            out.append(p_)
    return out


def srim_clear_outputs(srim_dir):
    """Delete the TRIM output files of an earlier run (so that only the
    files of the new run are copied). Returns the files that could not be
    deleted (Program Files: deleted later with administrator rights)."""
    left = []
    for p_ in srim_output_files(srim_dir):
        try:
            os.remove(p_)
        except OSError:
            left.append(p_)
    return left


def srim_wait_outputs(srim_dir, stopped=lambda: False, still=3.0,
                      timeout=600.0):
    """Wait until TRIM has finished writing its output files: no TRIM.exe
    running and the files (names, sizes, times) unchanged for 'still'
    seconds. Returns False if stopped or after 'timeout' seconds."""
    import subprocess
    import time
    t_end = time.time() + timeout
    last, since = None, time.time()
    while time.time() < t_end:
        if stopped():
            return False
        r_ = subprocess.run(["tasklist", "/fi", "imagename eq TRIM.exe"],
                            capture_output=True, text=True,
                            creationflags=0x08000000)
        snap = []
        for p_ in srim_output_files(srim_dir):
            try:
                st = os.stat(p_)
                snap.append((p_, st.st_size, st.st_mtime))
            except OSError:
                pass
        snap.sort()
        if "TRIM.exe" in r_.stdout or snap != last:
            last, since = snap, time.time()
        elif time.time() - since >= still:
            return True
        time.sleep(0.5)
    return False


def srim_copy_outputs(srim_dir, dest, prefix=""):
    """Copy TRIM.IN and all the TRIM output files (srim_output_files) to
    dest, with prefix before each name (the structure name, e.g.
    'Au_SiO2_RANGE.txt'). Copies (pysrim moves them, which fails in
    Program Files). Returns the names written."""
    import shutil
    done = []
    src_in = os.path.join(srim_dir, "TRIM.IN")
    for src in ([src_in] if os.path.isfile(src_in) else []) + \
            srim_output_files(srim_dir):
        name = prefix + os.path.basename(src)
        shutil.copy2(src, os.path.join(dest, name))
        done.append(name)
    return done


def srim_known(name):
    """(SRIM_OUTPUTS key, prefix) of a file name: the known TRIM file it
    is, also with a prefix ('Au_SiO2_RANGE.txt' -> ('RANGE.txt',
    'Au_SiO2_')); (None, '') if none. The longest key wins (RANGE_3D.txt
    before RANGE.txt is not an issue: suffixes must match)."""
    up = name.upper()
    best = None
    for k in SRIM_OUTPUTS:
        if up.endswith(k.upper()) and (best is None or len(k) > len(best)):
            best = k
    return (best, name[:len(name) - len(best)]) if best else (None, "")


class XCursor:
    """Vertical line dragged with the mouse on a matplotlib plot, with a
    dot on each curve and a box with x and the y of each curve at that x
    (interpolated; in log x, linearly in log). draw(series) after each
    redraw of the axes: series = [(label, x, y, colour, axes)]."""

    def __init__(self, canvas, ax, toolbar, xlabel="x"):
        self.canvas, self.ax, self.toolbar = canvas, ax, toolbar
        self.xlabel = xlabel
        self.x = None
        self.series = []
        self.line = self.text = None
        self.dots = []
        self.drag = False
        canvas.mpl_connect("button_press_event", self._press)
        canvas.mpl_connect("motion_notify_event", self._move)
        canvas.mpl_connect("button_release_event",
                           lambda e: setattr(self, "drag", False))

    def draw(self, series, on=True, xlabel=None):
        """New curves (the axes were cleared): line, dots and box again;
        the x is kept if it is still inside the curves."""
        self.series = [s for s in series if len(s[1])]
        self.line = self.text = None
        self.dots = []
        if xlabel:
            self.xlabel = xlabel
        if not on or not self.series:
            return
        lo = min(float(np.nanmin(s[1])) for s in self.series)
        hi = max(float(np.nanmax(s[1])) for s in self.series)
        if self.x is None or not lo <= self.x <= hi:
            self.x = (lo * hi) ** 0.5 if self.ax.get_xscale() == "log" \
                and lo > 0 else (lo + hi) / 2
        self.line = self.ax.axvline(self.x, color="#000000", ls=":",
                                    lw=1.2, zorder=2000)
        for lab, x, y, col, a_ in self.series:
            self.dots.append(a_.plot([], [], "o", ms=5, color=col,
                                     mec="#000000", mew=0.8,
                                     zorder=2001)[0])
        self.text = self.ax.text(
            0.01, 0.99, "", transform=self.ax.transAxes, ha="left",
            va="top", fontsize=8, family="monospace", zorder=2002,
            bbox=dict(boxstyle="round", fc="#ffffff", ec="#888888",
                      alpha=0.9))
        self.set(self.x)

    def value(self, x, xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        o = np.argsort(xs)
        xs, ys = xs[o], ys[o]
        if not len(xs) or x < xs[0] or x > xs[-1]:
            return None
        if self.ax.get_xscale() == "log" and xs[0] > 0 and x > 0:
            return float(np.interp(np.log(x), np.log(xs), ys))
        return float(np.interp(x, xs, ys))

    def set(self, x):
        self.x = x
        if self.line is None:
            return
        self.line.set_xdata([x, x])
        rows = [f"{self.xlabel} = {x:.5g}"]
        for (lab, xs, ys, col, a_), dot in zip(self.series, self.dots):
            v = self.value(x, xs, ys)
            if v is None:
                dot.set_data([], [])
                rows.append(f"{lab}: –")
            else:
                dot.set_data([x], [v])
                rows.append(f"{lab}: {v:.5g}")
        self.text.set_text("\n".join(rows))

    def _press(self, e):
        # any axes of the figure (a twin axes on top shares the x)
        if (self.line is None or e.button != 1 or self.toolbar.mode or
                e.xdata is None or
                e.inaxes not in self.canvas.figure.axes):
            return
        self.drag = True
        self.set(e.xdata)
        self.canvas.draw_idle()

    def _move(self, e):
        if not self.drag or e.xdata is None:
            return
        self.set(e.xdata)
        self.canvas.draw_idle()


# SRIM 'SR Module' (Stopping and Range tables): SR.IN in the SR Module
# folder, SRModule.exe writes the table in the output file named there
SR_UNITS = ["1 - eV / Angstrom", "2 - keV / micron", "3 - MeV / mm",
            "4 - keV / (ug/cm2)", "5 - MeV / (mg/cm2)", "6 - keV / (mg/cm2)",
            "7 - eV / (1E15 atoms/cm2)", "8 - L.S.S. reduced units"]


SR_OUT_NAME = "ionolum_SR.txt"


def sr_in_text(ion_z, ion_mass, phase, density, corr, elements, units,
               e_min, e_max, out_name=SR_OUT_NAME):
    """Text of SR.IN. elements = [(Z, name, stoichiometry, mass)]."""
    lines = ["---Stopping/Range Input Data (Number-format: Period = Decimal "
             "Point)",
             "---Output File Name", f'"{out_name}"',
             "---Ion(Z), Ion Mass(u)", f"{ion_z} {ion_mass:.4f}",
             "---Target Data: (Solid=0,Gas=1), Density(g/cm3), Compound "
             "Corr.", f"{phase} {density:.5g} {corr:g}",
             "---Number of Target Elements", f"{len(elements)}",
             "---Target Elements: (Z), Target name, Stoich, Target Mass(u)"]
    lines += [f'{z} "{nm}" {st:.6g} {m:.4f}' for z, nm, st, m in elements]
    lines += ["---Output Stopping Units (1-8)", f"{units}",
              "---Ion Energy : E-Min(keV), E-Max(keV)",
              f"{e_min:g} {e_max:g}"]
    return "\n".join(lines) + "\n"


def read_sr_output(path):
    """Table of an SR Module output file: dict of numpy arrays energy
    (keV), elec, nucl (stopping, in the units of the file), range, long,
    lat (Å). ValueError if no table is found."""
    with open(path, "rb") as f:
        text = f.read().decode("latin-1")
    e_mult = {"eV": 1e-3, "keV": 1.0, "MeV": 1e3, "GeV": 1e6}
    l_mult = {"A": 1.0, "um": 1e4, "µm": 1e4, "mm": 1e7, "m": 1e10,
              "km": 1e13}
    num = r"([-+\d.,]+(?:[eE][-+]?\d+)?)"
    rx = re.compile(r"^\s*" + num + r"\s*(eV|keV|MeV|GeV)\s+" + num +
                    r"\s+" + num + r"\s+" + num + r"\s*(A|um|µm|mm|m|km)\s+" +
                    num + r"\s*(A|um|µm|mm|m|km)\s+" + num +
                    r"\s*(A|um|µm|mm|m|km)\s*$")
    rows = []
    for ln in text.splitlines():
        m = rx.match(ln)
        if not m:
            continue
        g = [x.replace(",", ".") for x in m.groups()]
        rows.append((float(g[0]) * e_mult[g[1]], float(g[2]), float(g[3]),
                     float(g[4]) * l_mult[g[5]], float(g[6]) * l_mult[g[7]],
                     float(g[8]) * l_mult[g[9]]))
    if not rows:
        raise ValueError("no stopping table found in the file")
    a = np.array(rows)
    return {"energy": a[:, 0], "elec": a[:, 1], "nucl": a[:, 2],
            "range": a[:, 3], "long": a[:, 4], "lat": a[:, 5]}


def trim_in_of(folder, prefix=None):
    """TRIM.IN that goes with the output files of a folder: '<prefix>
    TRIM.IN' (structure name of the copied files) if a prefix is given and
    the file exists, else TRIM.IN in the folder or, for 'SRIM Outputs', in
    the SRIM folder above it, else the newest '*TRIM.IN' of the folder."""
    if prefix:
        p_ = os.path.join(folder, prefix + "TRIM.IN")
        if os.path.isfile(p_):
            return p_
    for d_ in (folder, os.path.dirname(os.path.normpath(folder))):
        p_ = os.path.join(d_, "TRIM.IN")
        if os.path.isfile(p_):
            return p_
    try:
        named = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.upper().endswith("TRIM.IN")]
    except OSError:
        named = []
    if named:
        return max(named, key=os.path.getmtime)
    return os.path.join(folder, "TRIM.IN")
