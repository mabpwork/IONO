"""SRIM tab of ionolum (TRIM with pysrim; sub-tabs Structure, Results, Compound dictionary and
Stopping & Range). Made from the SRIM tab of IBIST.py; it is now ionolum's own code.

build_srim_tab(root, nb, hooks): hooks["target"]() = layers of the target from elsewhere (here none:
the target layers are the ones added in Structure); it sets hooks["setion"] (ion from the
Experimental setup), hooks["srim_state"] / hooks["srim_restore"] (sessions) and hooks["srim_busy"].
"""

import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .srim_tools import (
    ATOM_ENERGY_FILE, LAYER_COLORS, N_AVOGADRO, PALETTE, SRIM_OUTPUTS, SR_OUT_NAME, SR_UNITS,
    TRIM_CALCS, TRIM_COLLISIONS, TRIM_PLOTS, XCursor, import_srim, isotope_masses, layer_density,
    read_atom_energy_table, read_atomdata, read_sr_output, read_srim_compounds, read_trim_in,
    read_trim_in_layers, run_elevated, sr_in_text, srim_clear_outputs, srim_copy_outputs,
    srim_energies, srim_known, srim_output_folder, srim_parse_comp, srim_table, srim_wait_outputs,
    trim_in_atom_energies, trim_in_lf, trim_in_of, trim_in_text, write_atom_energy_table)


def edit_cell(tree, iid, column, initial, on_done):
    """Small entry box over a Treeview cell; Enter/click away saves,
    Esc cancels."""
    bbox = tree.bbox(iid, column)
    if not bbox:
        return
    x, y, w, h = bbox
    ent = ttk.Entry(tree)
    ent.insert(0, initial)
    ent.select_range(0, "end")
    ent.place(x=x, y=y, width=w, height=h)
    ent.focus_set()
    done = [False]

    def finish(save):
        if done[0]:
            return
        done[0] = True
        text = ent.get()
        ent.destroy()
        if save and text.strip() != initial:
            on_done(text)
    ent.bind("<Return>", lambda e: finish(True))
    ent.bind("<KP_Enter>", lambda e: finish(True))
    ent.bind("<Escape>", lambda e: finish(False))
    ent.bind("<FocusOut>", lambda e: finish(True))


def build_srim_tab(root, nb, hooks):
    """The SRIM tab in the notebook nb. hooks: dict shared with the program; hooks["target"]()
    = layers of the target from elsewhere (here none)."""
    hooks.setdefault("target", lambda: [])

    # ===================== TAB: SRIM ========================================
    # TRIM simulations with pysrim. Sub-tab Structure: the target layers
    # (the ones of NDF > Config files, updated with them), the ion and the
    # TRIM options; the TRIM.IN file is shown on the right. Sub-tab Results:
    # the output files of a TRIM run and their plot on the right.
    tab_s = ttk.Frame(nb, padding=4)
    nb.add(tab_s, text="  SRIM  ")
    pan_s = ttk.PanedWindow(tab_s, orient="horizontal")
    pan_s.pack(fill="both", expand=True)
    sub_s = ttk.Notebook(pan_s, width=460)
    right_s = ttk.Frame(pan_s)
    pan_s.add(sub_s, weight=0)
    pan_s.add(right_s, weight=1)

    # ----------------------- SRIM sub-tab: Structure ------------------------
    tab_ss_outer = ttk.Frame(sub_s)
    sub_s.add(tab_ss_outer, text="  Structure  ")
    ss_canvas = tk.Canvas(tab_ss_outer, highlightthickness=0, bd=0,
                          bg=ttk.Style().lookup("TFrame", "background"))
    ss_vsb = ttk.Scrollbar(tab_ss_outer, orient="vertical",
                           command=ss_canvas.yview)
    ss_canvas.configure(yscrollcommand=ss_vsb.set)
    ss_vsb.pack(side="right", fill="y")
    ss_canvas.pack(side="left", fill="both", expand=True)
    tab_ss = ttk.Frame(ss_canvas, padding=8)
    ss_win = ss_canvas.create_window((0, 0), window=tab_ss, anchor="nw")
    tab_ss.bind("<Configure>", lambda e: ss_canvas.configure(
        scrollregion=ss_canvas.bbox("all")))
    ss_canvas.bind("<Configure>", lambda e: ss_canvas.itemconfigure(
        ss_win, width=e.width))

    def ss_wheel(e):
        if ss_canvas.yview() != (0.0, 1.0):
            ss_canvas.yview_scroll(int(-e.delta / 120) or
                                   (-1 if e.delta > 0 else 1), "units")
    tab_ss_outer.bind("<Enter>", lambda e: root.bind_all("<MouseWheel>",
                                                         ss_wheel))
    tab_ss_outer.bind("<Leave>", lambda e: root.unbind_all("<MouseWheel>"))

    sr = {k: tk.StringVar(value=v) for k, v in (
        ("ion", "He"), ("mass", ""), ("energy", "2000"), ("angle", "0"),
        ("nions", "1000"), ("calc", TRIM_CALCS[0]), ("seed", ""),
        ("bragg", "1.0"), ("autosave", "0"), ("description", "ionolum run"),
        ("plot", TRIM_PLOTS[5]), ("xmin", "0"), ("xmax", "0"),
        ("collisions", TRIM_COLLISIONS[0]), ("exyz", "0"),
        ("srimdir", r"C:\Program Files (x86)\SRIM"),
        ("outdir", ""), ("structname", ""))}
    sr_flags = {k: tk.BooleanVar(value=False) for k in (
        "ranges", "backscattered", "transmit", "sputtered")}

    def sr_row(parent, r, text, key, width=10, values=None):
        ttk.Label(parent, text=text).grid(row=r, column=0, sticky="w",
                                          pady=1)
        if values:
            w_ = ttk.Combobox(parent, textvariable=sr[key], values=values,
                              state="readonly", width=width)
        else:
            w_ = tk.Entry(parent, textvariable=sr[key], width=width,
                          justify="right")
        w_.grid(row=r, column=1, sticky="e", pady=1)
        return w_

    # ion
    lf_ion = ttk.LabelFrame(tab_ss, text="Ion (projectile)", padding=6)
    lf_ion.pack(fill="x")
    lf_ion.columnconfigure(0, weight=1)
    ent_ion = sr_row(lf_ion, 0, "Element", "ion", 8)
    ent_ion.config(justify="center")
    ttk.Label(lf_ion, text="Mass (amu)").grid(row=1, column=0, sticky="w",
                                              pady=1)
    cb_smass = ttk.Combobox(lf_ion, textvariable=sr["mass"], width=9,
                            justify="right")
    cb_smass.grid(row=1, column=1, sticky="e", pady=1)

    def on_sion(*_):
        """The mass follows the ion typed: most abundant isotope
        (periodictable; the others are in the list). Unknown -> empty."""
        masses = isotope_masses(sr["ion"].get())
        cb_smass.config(values=[f"{m:.4f}" for _, m, _ in masses])
        sr["mass"].set(f"{masses[0][1]:.4f}" if masses else "")
    sr["ion"].trace_add("write", on_sion)
    on_sion()
    sr_row(lf_ion, 2, "Energy (keV)", "energy")
    sr_row(lf_ion, 3, "Angle of incidence (°)", "angle")
    lbl_ion = ttk.Label(lf_ion, text="", foreground="#555555")
    lbl_ion.grid(row=4, column=0, columnspan=2, sticky="w")

    # calculation
    lf_calc = ttk.LabelFrame(tab_ss, text="TRIM calculation", padding=6)
    lf_calc.pack(fill="x", pady=(8, 0))
    lf_calc.columnconfigure(0, weight=1)
    sr_row(lf_calc, 0, "Number of ions", "nions")
    ttk.Label(lf_calc, text="Type of damage calculation").grid(
        row=1, column=0, columnspan=2, sticky="w")
    ttk.Combobox(lf_calc, textvariable=sr["calc"], values=TRIM_CALCS,
                 state="readonly").grid(row=2, column=0, columnspan=2,
                                        sticky="ew", pady=(0, 2))
    sr_row(lf_calc, 3, "Random seed (empty = random)", "seed")
    sr_row(lf_calc, 4, "Compound correction (Bragg)", "bragg")
    sr_row(lf_calc, 5, "Autosave at ion # (0 = no)", "autosave")
    ttk.Label(lf_calc, text="Description").grid(row=6, column=0,
                                                sticky="w", pady=1)
    tk.Entry(lf_calc, textvariable=sr["description"], width=24).grid(
        row=6, column=1, sticky="e", pady=1)
    ttk.Label(lf_calc, text="Plot during the run").grid(
        row=7, column=0, columnspan=2, sticky="w")
    ttk.Combobox(lf_calc, textvariable=sr["plot"], values=TRIM_PLOTS,
                 state="readonly").grid(row=8, column=0, columnspan=2,
                                        sticky="ew", pady=(0, 2))
    ttk.Label(lf_calc, text="Plot depths xmin, xmax (Å; 0 0 = all)").grid(
        row=9, column=0, sticky="w", pady=1)
    fr_px = ttk.Frame(lf_calc)
    fr_px.grid(row=9, column=1, sticky="e")
    for k_ in ("xmin", "xmax"):
        tk.Entry(fr_px, textvariable=sr[k_], width=7,
                 justify="right").pack(side="left", padx=(2, 0))

    # output files
    lf_out = ttk.LabelFrame(tab_ss, text="Output files", padding=6)
    lf_out.pack(fill="x", pady=(8, 0))
    lf_out.columnconfigure(0, weight=1)
    for i, (k_, txt) in enumerate((
            ("ranges", "Ion ranges (RANGE_3D.txt)"),
            ("backscattered", "Backscattered ions (BACKSCAT.txt)"),
            ("transmit", "Transmitted ions (TRANSMIT.txt)"),
            ("sputtered", "Sputtered atoms (SPUTTER.txt)"))):
        ttk.Checkbutton(lf_out, text=txt, variable=sr_flags[k_]).grid(
            row=i, column=0, columnspan=2, sticky="w")
    sr_row(lf_out, 4, "Collision details (COLLISON.txt)", "collisions", 18,
           TRIM_COLLISIONS)
    sr_row(lf_out, 5, "EXYZ.txt energy step (eV, 0 = no)", "exyz")

    # target layers (from NDF > Config files)
    lf_tl = ttk.LabelFrame(tab_ss, text="Target layers", padding=6)
    lf_tl.pack(fill="x", pady=(8, 0))
    tv_sl = ttk.Treeview(lf_tl, columns=("n", "name", "comp", "thick",
                                         "rho", "dens", "width", "gas"),
                         show="headings", height=5, selectmode="extended")
    for c_, t_, w_, an in (("n", "#", 24, "center"),
                           ("name", "Name", 60, "w"),
                           ("comp", "Composition", 120, "w"),
                           ("thick", "10¹⁵ at/cm²", 70, "e"),
                           ("rho", "ρ (g/cm³)", 60, "e"),
                           ("dens", "n (at/cm³)", 72, "e"),
                           ("width", "Width (Å)", 70, "e"),
                           ("gas", "Gas", 32, "center")):
        tv_sl.heading(c_, text=t_)
        tv_sl.column(c_, width=w_, anchor=an, stretch=(c_ == "comp"))
    tv_sl.tag_configure("bad", background="#f8c4c4")
    tv_sl.tag_configure("user", foreground="#1a5fb4")
    tv_sl.pack(fill="x")
    lbl_sl = ttk.Label(lf_tl, text="", foreground="#555555")
    lbl_sl.pack(anchor="w")
    fr_slb = ttk.Frame(lf_tl)
    fr_slb.pack(anchor="e", pady=(4, 0))
    fr_slb2 = ttk.Frame(lf_tl)
    fr_slb2.pack(side="bottom", anchor="e", pady=(4, 0))
    ttk.Button(fr_slb2, text="Load TRIM.IN…",
               command=lambda: srim_load_trim_in()).pack(side="left")
    ttk.Label(fr_slb2, text="(target layers + atom energies)",
              foreground="#555555").pack(side="left", padx=(4, 0))
    ttk.Button(fr_slb, text="Add SRIM layer",
               command=lambda: srim_add_layer()).pack(side="left")
    ttk.Button(fr_slb, text="Remove SRIM layer(s)",
               command=lambda: srim_remove_layers()).pack(side="left",
                                                          padx=4)
    ttk.Button(fr_slb, text="Reload layers",
               command=lambda: srim_refresh()).pack(side="left", padx=4)
    ttk.Button(fr_slb, text="Reset SRIM changes",
               command=lambda: (srim_rho.clear(), srim_ovr.clear(),
                                srim_gas.clear(), srim_extra.clear(),
                                srim_removed.clear(), srim_order.clear(),
                                srim_refresh())).pack(side="left")

    # target atoms: displacement, lattice and surface binding energies
    lf_te = ttk.LabelFrame(tab_ss, text="Target atoms (energies in eV)",
                           padding=6)
    lf_te.pack(fill="x", pady=(8, 0))
    tv_se = ttk.Treeview(lf_te, columns=("el", "ed", "lat", "surf"),
                         show="headings", height=4, selectmode="browse")
    for c_, t_ in (("el", "Element"), ("ed", "Displacement"),
                   ("lat", "Lattice binding"), ("surf", "Surface binding")):
        tv_se.heading(c_, text=t_)
        tv_se.column(c_, width=90, anchor="center")
    tv_se.pack(fill="x")
    ttk.Label(lf_te, text="Double-click a value to change it.",
              foreground="#555555").pack(anchor="w", pady=(4, 0))
    fr_teb = ttk.Frame(lf_te)
    fr_teb.pack(fill="x", pady=(4, 0))
    ttk.Button(fr_teb, text="From TRIM.IN…",
               command=lambda: atoms_from_trim_in()).pack(side="left")
    ttk.Button(fr_teb, text="Save as my defaults",
               command=lambda: atoms_save_defaults()).pack(side="left",
                                                           padx=4)
    ttk.Button(fr_teb, text="Reset to defaults",
               command=lambda: atoms_reset()).pack(side="left")
    lbl_te = ttk.Label(lf_te, text="", foreground="#555555",
                       wraplength=400, justify="left")
    lbl_te.pack(anchor="w", pady=(4, 0))
    atomdata = {"path": None, "data": {}}

    def atom_defaults(s_):
        """[E_d, lattice, surface] (text) of a new target element: your
        table (srim_atom_energies.txt) > surface binding = heat of
        sublimation of SRIM's Data/ATOMDATA (E_d 25, lattice 3) > 25 / 3 /
        2 eV (also when ATOMDATA has 0, gases)."""
        tab_ = read_atom_energy_table()
        if s_ in tab_:
            return ["%g" % v for v in tab_[s_]]
        p_ = os.path.join(sr["srimdir"].get().strip(), "Data", "ATOMDATA")
        if atomdata["path"] != p_:
            atomdata["path"] = p_
            try:
                atomdata["data"] = read_atomdata(p_)
            except OSError:
                atomdata["data"] = {}
        hs = atomdata["data"].get(s_, {}).get("heat_subl", 0.0)
        return ["25", "3", "%g" % hs if hs > 0 else "2"]

    def atoms_source_text():
        tab_ = read_atom_energy_table()
        atom_defaults("Si")                 # (reads ATOMDATA if needed)
        ok = bool(atomdata["data"])
        return ("Defaults of new elements: "
                + (f"your table ({len(tab_)} elements, "
                   f"{os.path.basename(ATOM_ENERGY_FILE)}), then " if tab_
                   else "")
                + ("surface binding = heat of sublimation of SRIM "
                   "Data/ATOMDATA (gases: 2), E_d 25, lattice 3."
                   if ok else "25 / 3 / 2 eV (ATOMDATA of SRIM not found)."))

    def atoms_from_trim_in():
        """E_d, lattice and surface energies of a TRIM.IN (e.g. saved by
        SRIM's TRIM Setup) -> the elements of the list that are in it."""
        p_ = filedialog.askopenfilename(
            title="TRIM.IN with the target atom energies",
            initialdir=sr["srimdir"].get().strip() or None,
            filetypes=[("TRIM input", "TRIM.IN *.IN *.in"),
                       ("All", "*.*")])
        if not p_:
            return
        try:
            got = trim_in_atom_energies(p_)
        except (OSError, ValueError) as e:
            messagebox.showerror("From TRIM.IN", f"{p_}:\n{e}")
            return
        used = [s_ for s_ in got if s_ in srim_atoms]
        for s_ in got:
            srim_atoms[s_] = ["%g" % v for v in got[s_]]
        srim_refresh()
        lbl_te.config(text=f"From {os.path.basename(p_)}: " + ", ".join(
            "%s %g/%g/%g" % (s_, *got[s_]) for s_ in got)
            + ("" if used else "  (none of these elements is in the "
                                "target now)"))

    def atoms_save_defaults():
        """The values of the list -> your table (merged with it): used for
        these elements from now on."""
        tab_ = read_atom_energy_table()
        for iid in tv_se.get_children():
            try:
                tab_[iid] = [float(x) for x in srim_atoms[iid]]
            except (KeyError, ValueError):
                pass
        try:
            write_atom_energy_table(tab_)
        except OSError as e:
            messagebox.showerror("Save as my defaults", str(e))
            return
        lbl_te.config(text=f"Saved in {globals()['ATOM_ENERGY_FILE']} "
                           f"({len(tab_)} "
                           f"elements). " + atoms_source_text())

    def atoms_reset():
        """The elements of the list back to the defaults."""
        for iid in tv_se.get_children():
            srim_atoms[iid] = atom_defaults(iid)
        srim_refresh()
        lbl_te.config(text=atoms_source_text())
    root.after(1000, lambda: lbl_te.config(text=atoms_source_text()))

    # run
    lf_run = ttk.LabelFrame(tab_ss, text="TRIM run", padding=6)
    lf_run.pack(fill="x", pady=(8, 0))
    lf_run.columnconfigure(1, weight=1)
    for r_, (k_, txt, btn, title) in enumerate((
            ("srimdir", "SRIM folder (TRIM.exe)", "SRIM folder…",
             "Folder of SRIM (with TRIM.exe)"),
            ("outdir", "Copy the results to", "Results folder…",
             "Folder for the TRIM output files"))):
        ttk.Label(lf_run, text=txt).grid(row=2 * r_, column=0,
                                         columnspan=3, sticky="w")
        tk.Entry(lf_run, textvariable=sr[k_]).grid(
            row=2 * r_ + 1, column=0, columnspan=2, sticky="ew", pady=1)
        ttk.Button(lf_run, text=btn,
                   command=lambda k_=k_, t_=title: (
                       lambda d_: d_ and sr[k_].set(d_))(
                       filedialog.askdirectory(title=t_))).grid(
            row=2 * r_ + 1, column=2, padx=(4, 0))
    # name of the structure: put before the names of the files copied from
    # the SRIM folder (e.g. Au_SiO2_RANGE.txt, Au_SiO2_TRIM.IN)
    fr_sn = ttk.Frame(lf_run)
    fr_sn.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(2, 0))
    ttk.Label(fr_sn, text="Structure name").pack(side="left")
    tk.Entry(fr_sn, textvariable=sr["structname"], width=16).pack(
        side="left", padx=4, fill="x", expand=True)
    # the prefix the copied files will get (name_element_energy_)
    lbl_sn = ttk.Label(lf_run, text="", foreground="#555555")
    lbl_sn.grid(row=5, column=0, columnspan=3, sticky="w")

    def sn_show(*_):
        p_ = srim_prefix()
        lbl_sn.config(text=f"Copied files: {p_}RANGE.txt, {p_}TRIM.IN, …")
    for k_ in ("structname", "ion", "energy"):
        sr[k_].trace_add("write", sn_show)
    fr_rb = ttk.Frame(lf_run)
    fr_rb.grid(row=6, column=0, columnspan=3, sticky="e", pady=(6, 0))
    ttk.Button(fr_rb, text="Save TRIM.IN…",
               command=lambda: srim_save_in()).pack(side="left", padx=4)
    trim_state = [None]      # state of the TRIM run going on (or None)
    b_trim = ttk.Button(fr_rb, text="Run TRIM",
                        command=lambda: messagebox.showinfo(
                            "Run TRIM", "TRIM is still running: press Stop "
                                        "TRIM first.") if trim_state[0]
                        else srim_run())
    b_trim.pack(side="left")
    # active only while TRIM runs
    b_trim_stop = ttk.Button(fr_rb, text="Stop TRIM",
                             command=lambda: srim_stop(), state="disabled")
    b_trim_stop.pack(side="left", padx=(4, 0))
    lbl_trim = ttk.Label(lf_run, text="", foreground="#555555",
                         wraplength=420)
    lbl_trim.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def srim_prefix():
        """Prefix of the copied files: structure name, ion element and
        energy, separated by '_' (e.g. 'Au_SiO2_He_2000keV_'); characters
        not allowed in file names removed; the parts that are empty are
        left out."""
        def clean(t_):
            return re.sub(r'[\\/:*?"<>|\s]', "", t_).strip("_")
        ion = clean(sr["ion"].get())
        ion = ion[:1].upper() + ion[1:].lower()
        en = clean(sr["energy"].get().replace(",", "."))
        parts = [clean(sr["structname"].get()), ion,
                 f"{en}keV" if en else ""]
        n_ = "_".join(p_ for p_ in parts if p_)
        return n_ + "_" if n_ else ""
    sn_show()

    # right pane of Structure: drawing of the target | TRIM.IN (editable)
    trimin_area = ttk.Frame(right_s, padding=8)
    pan_ti = ttk.PanedWindow(trimin_area, orient="horizontal")
    pan_ti.pack(fill="both", expand=True)
    fr_sdraw = ttk.Frame(pan_ti)
    fr_tin = ttk.Frame(pan_ti, padding=(8, 0, 0, 0))
    pan_ti.add(fr_sdraw, weight=3)
    pan_ti.add(fr_tin, weight=2)
    ttk.Label(fr_sdraw, text="Target structure (SRIM)",
              font=("Segoe UI", 10, "bold")).pack(anchor="w")
    sstruct_cv = tk.Canvas(fr_sdraw, bg="#ffffff", highlightthickness=1,
                           highlightbackground="#cccccc")
    sstruct_cv.pack(fill="both", expand=True, pady=(4, 0))
    fr_tih = ttk.Frame(fr_tin)
    fr_tih.pack(fill="x")
    ttk.Label(fr_tih, text="TRIM.IN", font=("Segoe UI", 10, "bold")).pack(
        side="left")
    v_trim_auto = tk.BooleanVar(value=True)
    ttk.Checkbutton(fr_tih, text="Auto-generate TRIM.IN",
                    variable=v_trim_auto,
                    command=lambda: srim_show_in()).pack(side="right")
    txt_trimin = ScrolledText(fr_tin, font=("Courier", 9), wrap="none",
                              width=44, undo=True)
    txt_trimin.pack(fill="both", expand=True, pady=(4, 0))
    ttk.Label(fr_tin, text="Editable: typing here switches off the "
                           "auto-generation, and Save / Run use this "
                           "text.", foreground="#555555",
              wraplength=320).pack(anchor="w", pady=(2, 0))
    trim_gen = [""]          # last TRIM.IN generated

    def on_trimin_key(_e=None):
        if v_trim_auto.get() and \
                txt_trimin.get("1.0", "end-1c") != trim_gen[0]:
            v_trim_auto.set(False)
    txt_trimin.bind("<KeyRelease>", on_trimin_key)

    # layer keys: int = index of the Config files layer, 'xN' = layer
    # added here (SRIM only)
    srim_rho = {}            # layer key -> density (g/cm3) typed
    srim_gas = {}            # layer key -> True if gas
    srim_ovr = {}            # layer key -> {name, comp, thick, width}
    srim_extra = {}          # key 'xN' -> {} of the layers added here
    srim_removed = set()     # Config files layers removed here
    srim_order = []          # order of the layer keys
    srim_atoms = {}          # element -> [E_d, lattice, surface] (eV)
    srim_lays = []           # layers shown: dict (see srim_refresh)
    srim_pending = [None]

    def srim_key(iid):
        return int(iid) if iid.isdigit() else iid

    def srim_num(key, kind=float, low=None):
        """Value of a SRIM box; ValueError with the name if not valid."""
        try:
            v_ = kind(float(sr[key].get().strip().replace(",", ".")))
        except ValueError:
            raise ValueError(f"'{sr[key].get()}' is not a valid value "
                             f"({key}).")
        if low is not None and v_ < low:
            raise ValueError(f"{key} must be ≥ {low}.")
        return v_

    def srim_refresh():
        """Layers and atoms from NDF > Config files, then TRIM.IN."""
        srim_pending[0] = None
        try:
            target = hooks["target"]()
        except Exception as e:
            lbl_sl.config(text=f"Layers of Config files not read: {e}",
                          foreground="#c62828")
            return
        sel_ = tv_sl.selection()
        srim_lays.clear()
        order = []
        # order of the layers: the one kept here, new Config files layers
        # after the previous one, removed / missing ones left out
        n_cf = len(target)
        keys = [k for k in srim_order
                if k in srim_extra or (isinstance(k, int) and k < n_cf
                                       and k not in srim_removed)]
        for i in range(n_cf):
            if i in srim_removed or i in keys:
                continue
            prev = [j for j, k in enumerate(keys)
                    if isinstance(k, int) and k < i]
            keys.insert(prev[-1] + 1 if prev else 0, i)
        srim_order[:] = keys
        empty = {"elements": [], "molecules": [], "thickness": 0}
        for i in keys:
            lay = empty if i in srim_extra else target[i]
            els = {}
            for e in lay["elements"]:
                # SRIM element: '29Si' -> Si, D/T -> H, any case
                s_ = re.sub(r"^\d+", "", str(e[0]).strip())
                s_ = s_[:1].upper() + s_[1:].lower()
                s_ = "H" if s_ in ("D", "T") else s_
                try:
                    c_ = float(e[1])
                except (TypeError, ValueError):
                    continue
                if s_ and c_ > 0:
                    els[s_] = els.get(s_, 0.0) + c_
            ov = srim_ovr.get(i, {})
            mol_list = lay["molecules"]
            if "comp" in ov:                   # changed here
                els, mol_list = dict(ov["comp"]), []
            for s_ in els:
                if s_ not in order:
                    order.append(s_)
            try:
                n_, rho, m_ = layer_density(
                    list(els.items()), mol_list,
                    None if "comp" in ov else lay.get("rho"))
            except Exception:
                n_ = rho = m_ = None
            if i in srim_rho and m_:
                rho = srim_rho[i]
                n_ = rho * N_AVOGADRO / m_ / 1e22
            th = float(ov.get("thick", lay["thickness"]) or 0)
            if "width" in ov:
                width = ov["width"]
                if n_:
                    th = width * n_ / 10.0
            else:
                width = th * 10.0 / n_ if n_ and th > 0 else None
            mols = " + ".join(str(m[0]) for m in mol_list if m[0])
            dname = (f"Added {i[1:]}" if i in srim_extra
                     else f"Layer {i + 1}")
            srim_lays.append({"key": i, "els": els, "thick": th, "rho": rho,
                              "n": n_, "m": m_,
                              "width": width, "mols": mols,
                              "name": ov.get("name", dname),
                              "user": (bool(ov) or i in srim_rho
                                       or i in srim_extra),
                              "gas": srim_gas.get(i, False)})
        tv_sl.delete(*tv_sl.get_children())
        for i, L in enumerate(srim_lays):
            comp = " ".join(f"{s_} {c:.3g}" for s_, c in L["els"].items())
            if L["mols"]:
                comp = f"{L['mols']} ({comp})"
            tags = (() if L["width"] else ("bad",)) + \
                (("user",) if L["user"] else ())
            tv_sl.insert("", "end", iid=str(L["key"]), tags=tags, values=(
                i + 1, L["name"], comp or "(empty)", f"{L['thick']:.4g}",
                f"{L['rho']:.4g}" if L["rho"] else "?",
                f"{L['n'] * 1e22:.4e}" if L["n"] else "?",
                f"{L['width']:.5g}" if L["width"] else "?",
                "☑" if L["gas"] else "☐"))
        tv_sl.config(height=max(2, min(8, len(srim_lays))))
        n_rm = len([k for k in srim_removed if k < n_cf])
        extra_ = (f" ({len(srim_extra)} added, {n_rm} removed here)"
                  if srim_extra or n_rm else "")
        lbl_sl.config(text=f"{len(srim_lays)} layer(s){extra_}, total "
                           f"{sum(L['thick'] for L in srim_lays):.6g} "
                           f"10¹⁵ at/cm²", foreground="#555555")
        tv_se.delete(*tv_se.get_children())
        for s_ in order:
            if s_ not in srim_atoms:
                srim_atoms[s_] = atom_defaults(s_)
            vals = srim_atoms[s_]
            tv_se.insert("", "end", iid=s_, values=(s_, *vals))
        tv_se.config(height=max(2, min(8, len(order))))
        keep = [s_ for s_ in sel_ if tv_sl.exists(s_)]
        if keep:
            tv_sl.selection_set(keep)
        srim_show_in()

    def srim_add_layer():
        """New layer (SRIM only) after the last selected one, or at the
        end; its composition and thickness are typed in the list."""
        n_ = 1
        while f"x{n_}" in srim_extra:
            n_ += 1
        k_ = f"x{n_}"
        srim_extra[k_] = {}
        sel_ = [srim_key(s_) for s_ in tv_sl.selection()]
        pos = [j for j, k in enumerate(srim_order) if k in sel_]
        srim_order.insert(pos[-1] + 1 if pos else len(srim_order), k_)
        srim_refresh()
        tv_sl.selection_set(k_)
        tv_sl.see(k_)

    def srim_load_trim_in():
        """Target of a TRIM.IN -> the Target layers list: the layers of
        Config files are removed here (SRIM only; 'Reset SRIM changes'
        brings them back) and the TRIM.IN layers are added as SRIM layers
        (name, composition, width Å, ρ, gas); the atom energies of the file
        too."""
        p_ = filedialog.askopenfilename(
            title="TRIM.IN (target to load)",
            initialdir=(sr["outdir"].get().strip() or
                        sr["srimdir"].get().strip() or None),
            filetypes=[("TRIM input", "*TRIM.IN *.IN *.in"),
                       ("All files", "*.*")])
        if not p_:
            return
        tl = read_trim_in_layers(p_)
        if not tl:
            messagebox.showerror("Load TRIM.IN",
                                 f"No target layers found in\n{p_}")
            return
        if srim_lays and not messagebox.askyesno(
                "Load TRIM.IN",
                f"Replace the {len(srim_lays)} target layer(s) of SRIM by "
                f"the {len(tl)} layer(s) of\n{p_}?\n\n(Config files is not "
                f"changed: 'Reset SRIM changes' brings its layers back.)"):
            return
        try:
            n_cf = len(hooks["target"]())
        except Exception:
            n_cf = 0
        for k_ in list(srim_extra):
            for d_ in (srim_ovr, srim_rho, srim_gas):
                d_.pop(k_, None)
        srim_extra.clear()
        srim_removed.clear()
        srim_removed.update(range(n_cf))
        srim_order.clear()
        for k, L in enumerate(tl, start=1):
            k_ = f"x{k}"
            srim_extra[k_] = {}
            tot = sum(L["comp"].values()) or 1.0
            srim_ovr[k_] = {"name": L["name"] or f"Layer {k}",
                            "comp": {s_: c / tot
                                     for s_, c in L["comp"].items()},
                            "width": float(L["width"])}
            srim_rho[k_] = float(L["density"])
            srim_gas[k_] = bool(L.get("gas"))
            srim_order.append(k_)
        note = ""
        try:
            for s_, e_ in trim_in_atom_energies(p_).items():
                srim_atoms[s_] = ["%g" % v for v in e_]
            note = " Atom energies of the file applied."
        except (OSError, ValueError):
            note = " (no atom energies in the file: defaults kept)"
        srim_refresh()
        lbl_sl.config(text=f"{len(tl)} layer(s) loaded from "
                           f"{os.path.basename(p_)} (SRIM only).{note}",
                      foreground="#2e7d32")

    def srim_remove_layers(_e=None):
        """Remove the selected layers (SRIM only: 'Reset SRIM changes'
        brings back the ones of Config files)."""
        sel_ = [srim_key(s_) for s_ in tv_sl.selection()]
        if not sel_:
            messagebox.showinfo("Target layers",
                                "Select the layers to remove (Ctrl/Shift-"
                                "click selects several).")
            return "break"
        for k_ in sel_:
            if k_ in srim_extra:
                srim_extra.pop(k_)
                for d_ in (srim_ovr, srim_rho, srim_gas):
                    d_.pop(k_, None)
            else:
                srim_removed.add(k_)
        srim_order[:] = [k for k in srim_order if k not in sel_]
        tv_sl.selection_set(())
        srim_refresh()
        return "break"
    tv_sl.bind("<Delete>", srim_remove_layers)

    def srim_schedule():
        """Called when the layers of NDF > Config files change."""
        if srim_pending[0] is None:
            srim_pending[0] = root.after(300, srim_refresh)
    hooks["srim"] = srim_schedule

    def cf_srim_lays():
        """Layers of NDF › Config files as SRIM layers (as srim_refresh,
        without the changes made in SRIM › Structure): elements, ρ (from
        the molecules, else pure elements), width (Å) = thickness × 10 / n,
        not gas. New elements get their default atom energies."""
        out = []
        for i, lay in enumerate(hooks["target"]()):
            els = {}
            for e in lay["elements"]:
                s_ = re.sub(r"^\d+", "", str(e[0]).strip())
                s_ = s_[:1].upper() + s_[1:].lower()
                s_ = "H" if s_ in ("D", "T") else s_
                try:
                    c_ = float(e[1])
                except (TypeError, ValueError):
                    continue
                if s_ and c_ > 0:
                    els[s_] = els.get(s_, 0.0) + c_
                    if s_ not in srim_atoms:
                        srim_atoms[s_] = atom_defaults(s_)
            n_, rho, _ = layer_density(list(els.items()), lay["molecules"],
                                       lay.get("rho"))
            th = float(lay["thickness"] or 0)
            mols = " + ".join(str(m[0]) for m in lay["molecules"] if m[0])
            out.append({"els": els, "rho": rho, "n": n_, "thick": th,
                        "width": th * 10.0 / n_ if n_ and th > 0 else None,
                        "gas": False,
                        "name": mols or "Layer %d" % (i + 1)})
        return out

    def srim_build(lays=None):
        """pysrim TRIM object of the boxes (layers: lays, default the list
        of SRIM › Structure); ValueError with a message."""
        srim_lays_ = srim_lays if lays is None else lays
        try:
            srim = import_srim()
        except Exception as e:
            raise ValueError(f"pysrim cannot be imported:\n{e}")
        sym = sr["ion"].get().strip()
        sym = sym[:1].upper() + sym[1:].lower()
        if not isotope_masses(sym):
            raise ValueError(f"'{sr['ion'].get()}' is not an element.")
        mass = sr["mass"].get().strip()
        energy = srim_num("energy", low=1e-9)
        ion = srim.Ion(sym, energy=energy * 1e3,
                       mass=float(mass) if mass else None)
        layers = []
        for i, L in enumerate(srim_lays_):
            if not L["els"] or not L["width"] or not L["rho"]:
                raise ValueError(f"Layer {i + 1}: composition, thickness "
                                 f"or density missing.")
            tot = sum(L["els"].values())
            els = {}
            for s_, c in L["els"].items():
                ed, lat, surf = (float(x) for x in srim_atoms[s_])
                els[s_] = {"stoich": c / tot, "E_d": ed, "lattice": lat,
                           "surface": surf}
            layers.append(srim.Layer(els, density=L["rho"],
                                     width=L["width"],
                                     phase=1 if L["gas"] else 0,
                                     name=L["name"].replace('"', "")))
        if not layers:
            raise ValueError("The target has no layers.")
        kw = dict(description=sr["description"].get().strip().replace(
                      '"', "") or "ionolum run",
                  plot_mode=int(sr["plot"].get()[0]),
                  plot_xmin=srim_num("xmin", low=0),
                  plot_xmax=srim_num("xmax", low=0),
                  collisions=int(sr["collisions"].get()[0]),
                  exyz=srim_num("exyz", int, 0),
                  angle_ions=srim_num("angle"),
                  bragg_correction=srim_num("bragg"))
        for k_, v_ in sr_flags.items():
            kw[k_] = int(v_.get())
        if sr["seed"].get().strip():
            kw["random_seed"] = srim_num("seed", int, 0)
        trim = srim.TRIM(srim.Target(layers), ion,
                         calculation=int(sr["calc"].get()[0]),
                         number_ions=srim_num("nions", int, 1), **kw)
        # pysrim accepts only 0/1 for autosave (TRIM: save every N ions)
        # and takes 'sputtered' from 'ranges': set them directly
        trim.settings._settings["autosave"] = srim_num("autosave", int, 0)
        trim.settings._settings["sputtered"] = int(
            sr_flags["sputtered"].get())
        return trim

    def srim_show_in(*_):
        """TRIM.IN of the current settings (or what is wrong)."""
        if hooks.get("srim_busy"):
            return
        sym = sr["ion"].get().strip()
        isos = isotope_masses(sym)
        ent_ion.config(bg="#ffffff" if isos else "#f28b82")
        lbl_ion.config(text=f"Most abundant isotope: {isos[0][0]}"
                            f"{sym[:1].upper() + sym[1:].lower()} "
                            f"({isos[0][1]:.4f} amu)" if isos else "")
        srim_draw()
        if not v_trim_auto.get():      # edited by hand: kept as it is
            return
        try:
            text = trim_in_text(srim_build())
        except Exception as e:
            text = f"TRIM.IN cannot be written:\n{e}"
        trim_gen[0] = text
        if txt_trimin.get("1.0", "end-1c") != text:
            txt_trimin.delete("1.0", "end")
            txt_trimin.insert("1.0", text)
            txt_trimin.edit_reset()

    def srim_input():
        """TRIM.IN to save / run: the text of the box if edited by hand,
        else generated from the settings."""
        if v_trim_auto.get():
            return trim_in_text(srim_build())
        t_ = txt_trimin.get("1.0", "end-1c")
        if not t_.strip():
            raise ValueError("The TRIM.IN box is empty.")
        return trim_in_lf(t_)

    def srim_draw(_e=None):
        """Layers side by side (the ion enters from the left), width
        proportional to the thickness in Å (at least MIN_W pixels)."""
        cv = sstruct_cv
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 80 or h < 80 or not srim_lays:
            return
        MIN_W = 34
        x0, x1 = 90, w - 20
        y0, y1 = 30, h - 50
        ws_ = [L["width"] or 0 for L in srim_lays]
        tot = sum(ws_)
        n = len(srim_lays)
        free = max(0, (x1 - x0) - MIN_W * n)
        px = [MIN_W + (free * wd / tot if tot > 0 else free / n)
              for wd in ws_]
        sel_ = tv_sl.selection()
        x, depth, last_lab = x0, 0.0, -100
        cv.create_text(x0, y1 + 14, text="0", font=("Segoe UI", 8),
                       fill="#555555")
        for i, (L, pw) in enumerate(zip(srim_lays, px)):
            col = LAYER_COLORS[i % len(LAYER_COLORS)]
            cur = str(L["key"]) in sel_
            r_ = cv.create_rectangle(x, y0, x + pw, y1, fill=col,
                                     outline="#000000",
                                     width=3 if cur else 1)
            comp = " ".join(f"{s_}{c:.3g}" for s_, c in L["els"].items())
            wtxt = f"{L['width']:.4g} Å" if L["width"] else "? Å"
            t_ = cv.create_text(x + pw / 2, (y0 + y1) / 2, angle=90,
                                text=f"{L['name']}   {L['mols'] or comp}"
                                     f"   {wtxt}"
                                     f"{'   (gas)' if L['gas'] else ''}",
                                font=("Segoe UI", 9))
            for it in (r_, t_):
                cv.tag_bind(it, "<Button-1>", lambda e, k=str(L["key"]): (
                    tv_sl.selection_set(k), tv_sl.see(k)))
            x += pw
            depth += L["width"] or 0
            if x - last_lab > 45 or i == n - 1:
                cv.create_text(x, y1 + 14, font=("Segoe UI", 8),
                               fill="#555555", text=f"{depth:.4g}")
                last_lab = x
        cv.create_text((x0 + x1) / 2, y1 + 32, font=("Segoe UI", 9),
                       text="depth (Å) →   (click a layer to select it)",
                       fill="#555555")
        # the ion: arrow from the left, at the angle of incidence
        try:
            ang = float(sr["angle"].get().replace(",", "."))
        except ValueError:
            ang = 0.0
        import math
        yc = (y0 + y1) / 2
        L_ = 70
        dx = L_ * math.cos(math.radians(ang))
        dy = L_ * math.sin(math.radians(ang))
        # beam: 10 parallel arrows over the middle of the target face
        N_ARR = 10
        span = (y1 - y0) * 0.6
        for k in range(N_ARR):
            ya = yc - span / 2 + span * k / (N_ARR - 1)
            cv.create_line(x0 - dx, ya - dy, x0, ya, arrow="last", width=2,
                           fill="#c62828")
        cv.create_text(x0 - 45, max(10, yc - span / 2 - 14 - max(dy, 0)),
                       fill="#c62828",
                       font=("Segoe UI", 9, "bold"),
                       text=f"{sr['ion'].get().strip()} "
                            f"{sr['energy'].get().strip()} keV")
    sstruct_cv.bind("<Configure>", srim_draw)
    tv_sl.bind("<<TreeviewSelect>>", srim_draw)
    for v_ in list(sr.values()) + list(sr_flags.values()):
        v_.trace_add("write", srim_show_in)

    def on_sl_double(e):
        """Double-click on a layer cell: change it here (SRIM only; an
        empty value goes back to the one of Config files)."""
        iid = tv_sl.identify_row(e.y)
        col = tv_sl.identify_column(e.x)
        if not iid:
            return "break"
        i = srim_key(iid)
        key = {"#2": "name", "#3": "comp", "#4": "thick", "#5": "rho",
               "#6": "dens", "#7": "width"}.get(col)
        if col == "#8":
            srim_gas[i] = not srim_gas.get(i, False)
            srim_refresh()
            return "break"
        if not key:
            return "break"
        L = next(L for L in srim_lays if L["key"] == i)
        initial = {"name": L["name"],
                   "comp": " ".join(f"{s_} {c:.4g}"
                                    for s_, c in L["els"].items()),
                   "thick": f"{L['thick']:.6g}",
                   "rho": f"{L['rho']:.5g}" if L["rho"] else "",
                   "dens": f"{L['n'] * 1e22:.5e}" if L["n"] else "",
                   "width": f"{L['width']:.6g}" if L["width"] else ""}[key]

        def done(text):
            t_ = text.strip()
            ov = srim_ovr.setdefault(i, {})
            try:
                if not t_:                     # back to Config files
                    ov.pop(key, None)
                    if key in ("rho", "dens"):
                        srim_rho.pop(i, None)
                elif key == "dens":            # n -> rho = n·M / N_A
                    v_ = float(t_.replace(",", "."))
                    if v_ <= 0:
                        raise ValueError("must be > 0")
                    if v_ < 1e4:               # typed in 1e22 at/cm3
                        v_ *= 1e22
                    if not L["m"]:
                        raise ValueError("the composition of the layer is "
                                         "not known")
                    srim_rho[i] = v_ * L["m"] / N_AVOGADRO
                elif key == "name":
                    ov["name"] = t_.replace('"', "")
                elif key == "comp":
                    ov["comp"] = srim_parse_comp(t_)
                else:
                    v_ = float(t_.replace(",", "."))
                    if v_ <= 0:
                        raise ValueError("must be > 0")
                    if key == "rho":
                        srim_rho[i] = v_
                    else:                      # thickness or width
                        ov[key] = v_
                        ov.pop("width" if key == "thick" else "thick",
                               None)
            except ValueError as e_:
                messagebox.showwarning("Target layers",
                                       f"'{text}' not accepted: {e_}")
                return
            if not ov:
                srim_ovr.pop(i, None)
            srim_refresh()
        edit_cell(tv_sl, iid, col, initial, done)
        return "break"
    tv_sl.bind("<Double-1>", on_sl_double)

    def on_se_double(e):
        iid = tv_se.identify_row(e.y)
        col = tv_se.identify_column(e.x)
        if not iid or col not in ("#2", "#3", "#4"):
            return "break"
        k_ = int(col[1:]) - 2

        def done(text):
            try:
                float(text.replace(",", "."))
            except ValueError:
                messagebox.showwarning("Target atoms",
                                       f"'{text}' is not a number.")
                return
            srim_atoms[iid][k_] = text.strip().replace(",", ".")
            srim_refresh()
        edit_cell(tv_se, iid, col, srim_atoms[iid][k_], done)
        return "break"
    tv_se.bind("<Double-1>", on_se_double)

    def srim_save_in():
        try:
            text = srim_input()
        except Exception as e:
            messagebox.showerror("Save TRIM.IN", str(e))
            return
        path = filedialog.asksaveasfilename(
            title="Save TRIM.IN", initialfile="TRIM.IN",
            initialdir=sr["srimdir"].get().strip() or None,
            filetypes=[("TRIM input", "*.IN"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="latin-1", newline="\r\n") as f:
                f.write(text)
        except Exception as e:
            messagebox.showerror("Save TRIM.IN", f"Could not write "
                                                 f"{path}:\n{e}")
            return
        lbl_trim.config(text=f"Saved {path}")

    def srim_run():
        """Write TRIM.IN (and TRIMAUTO, to run without questions) in the
        SRIM folder, run TRIM.exe in a thread, copy the output files to the
        results folder and show them in Results."""
        import threading
        import subprocess
        d_ = sr["srimdir"].get().strip()
        exe = os.path.join(d_, "TRIM.exe")
        if not os.path.isfile(exe):
            messagebox.showerror("Run TRIM", f"TRIM.exe not found in "
                                             f"'{d_}'. Choose the SRIM "
                                             f"folder.")
            return
        out = sr["outdir"].get().strip()
        if out and not os.path.isdir(out):
            messagebox.showerror("Run TRIM", f"'{out}' is not a folder.")
            return
        try:
            text = srim_input()
        except Exception as e:
            messagebox.showerror("Run TRIM", str(e))
            return
        # TRIM.IN and TRIMAUTO (run without questions) go to the SRIM
        # folder. If it cannot be written (Program Files), they are written
        # in a temporary folder and copied there with administrator rights,
        # in the same elevated command that runs TRIM.
        # the output files of an earlier run are deleted first, so that
        # only the new ones are copied; those that cannot be deleted here
        # (Program Files) are deleted by the elevated command
        left = srim_clear_outputs(d_)
        stage = None
        try:
            if left:
                raise PermissionError
            for name, txt_ in (("TRIM.IN", text), ("TRIMAUTO", "1")):
                with open(os.path.join(d_, name), "w", encoding="latin-1",
                          newline="\r\n") as f:
                    f.write(txt_)
        except PermissionError:
            import tempfile
            stage = tempfile.mkdtemp(prefix="ibist_trim_")
            for name, txt_ in (("TRIM.IN", text), ("TRIMAUTO", "1")):
                with open(os.path.join(stage, name), "w", encoding="latin-1",
                          newline="\r\n") as f:
                    f.write(txt_)
        except Exception as e:
            messagebox.showerror("Run TRIM", str(e))
            return
        state = {"error": None, "admin": False, "proc": None,
                 "handle": None}
        trim_state[0] = state

        def set_handle(h):
            state["handle"] = h

        def work():
            try:
                if stage:
                    state["admin"] = True
                    # 'del' joined with '&' (a failure does not stop the
                    # rest), then copy and run joined with '&&'
                    cmd = "".join(f'del /f /q "{p_}" 2>nul & '
                                  for p_ in left) + " && ".join(
                        [f'copy /y "{os.path.join(stage, n_)}" '
                         f'"{os.path.join(d_, n_)}" >nul'
                         for n_ in ("TRIM.IN", "TRIMAUTO")] +
                        [f'cd /d "{d_}"', "TRIM.exe"])
                    run_elevated("cmd.exe", f"/c {cmd}", d_, show=7,
                                 on_start=set_handle)
                else:
                    try:
                        state["proc"] = subprocess.Popen([exe], cwd=d_)
                        state["proc"].wait()
                    except OSError as e:
                        if getattr(e, "winerror", None) != 740:
                            raise
                        # TRIM.exe asks for administrator rights
                        state["admin"] = True
                        run_elevated(exe, "", d_, on_start=set_handle)
                state["handle"] = None
                # copy only when TRIM has finished writing the files
                state["waiting"] = True
                state["stable"] = srim_wait_outputs(
                    d_, lambda: state.get("stopped"))
                if out and not state.get("stopped"):
                    state["copied"] = srim_copy_outputs(d_, out,
                                                        srim_prefix())
            except Exception as e:
                state["error"] = e
            state["handle"] = None
            state["done"] = True

        def poll():
            if state is not trim_state[0]:   # stopped by the user
                return
            if not state.get("done"):
                if state.get("waiting"):
                    lbl_trim.config(text="TRIM ended; waiting until its "
                                         "output files are written…")
                root.after(500, poll)
                return
            trim_state[0] = None
            hooks["srim_busy"] = False
            b_trim_stop.state(["disabled"])
            if state["error"]:
                e_ = state["error"]
                msg = ("cancelled (administrator rights refused)"
                       if getattr(e_, "winerror", None) == 1223 else e_)
                lbl_trim.config(text=f"TRIM failed: {msg}")
                return
            res = out or srim_output_folder(d_)
            adm = " (as administrator)" if state["admin"] else ""
            n_cp = (f", {len(state.get('copied', []))} file(s) copied"
                    if out else "")
            warn = ("" if state.get("stable") else
                    " — the files were still changing (or TRIM.exe still "
                    "runs): use Reload in Results to copy them again")
            lbl_trim.config(text=f"TRIM finished{adm}; results in "
                                 f"{res}{n_cp}{warn}")
            v_sres.set(res)
            sres_scan()
            sub_s.select(tab_sr)
        hooks["srim_busy"] = True
        b_trim_stop.state(["!disabled"])
        lbl_trim.config(text="TRIM running… (if TRIM shows an error, close "
                             "its window or press Stop TRIM)")
        threading.Thread(target=work, daemon=True).start()
        root.after(500, poll)

    def srim_stop():
        """Stop TRIM (e.g. blocked on an error window) and free the Run
        button. An elevated TRIM may refuse to be stopped from here: then
        its window must be closed by hand, but the button is freed."""
        import subprocess
        state = trim_state[0]
        if state is None:
            return
        state["stopped"] = True
        msg = "TRIM stopped."
        try:
            if state["proc"] is not None and state["proc"].poll() is None:
                state["proc"].kill()
            elif state["handle"]:
                import ctypes
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                k32.TerminateProcess.argtypes = [ctypes.c_void_p,
                                                 ctypes.c_uint]
                k32.TerminateProcess(state["handle"], 1)
                # with the elevated cmd /c, TRIM.exe is a child of cmd
                subprocess.run(["taskkill", "/f", "/t", "/im", "TRIM.exe"],
                               capture_output=True,
                               creationflags=0x08000000)
                r_ = subprocess.run(["tasklist", "/fi",
                                     "imagename eq TRIM.exe"],
                                    capture_output=True, text=True,
                                    creationflags=0x08000000)
                if "TRIM.exe" in r_.stdout:
                    msg = ("TRIM runs as administrator and could not be "
                           "stopped from here: close its window (and the "
                           "error message) by hand.")
        except Exception as e:
            msg = (f"TRIM could not be stopped from here ({e}); close its "
                   f"window by hand.")
        trim_state[0] = None
        hooks["srim_busy"] = False
        b_trim_stop.state(["disabled"])
        lbl_trim.config(text=msg)
        srim_show_in()

    # ----------------------- SRIM sub-tab: Results --------------------------
    tab_sr = ttk.Frame(sub_s, padding=8)
    sub_s.add(tab_sr, text="  Results  ")
    fr_sd = ttk.Frame(tab_sr)
    fr_sd.pack(fill="x")
    ttk.Label(fr_sd, text="Folder:", font=("Segoe UI", 9, "bold")).pack(
        side="left")
    v_sres = tk.StringVar(value="")
    tk.Entry(fr_sd, textvariable=v_sres).pack(side="left", padx=4,
                                              fill="x", expand=True)

    def sres_browse():
        d_ = filedialog.askdirectory(title="Folder with SRIM output files",
                                     initialdir=v_sres.get().strip() or None)
        if d_:
            v_sres.set(d_)
            sres_scan()
    ttk.Button(fr_sd, text="Output folder…", command=sres_browse).pack(
        side="left")
    ttk.Button(fr_sd, text="Reload", command=lambda: sres_reload()).pack(
        side="left", padx=(4, 0))
    lbl_sres = ttk.Label(tab_sr, text="Reload copies again all the TRIM "
                                      "output files of the SRIM folder to "
                                      "the results folder (Structure › TRIM "
                                      "run) and lists them.",
                         foreground="#555555", wraplength=420)
    lbl_sres.pack(anchor="w", pady=(2, 0))

    def sres_reload():
        """Copy again TRIM.IN and all the output files of the SRIM folder
        to the results folder and show them (without a results folder:
        the files of the SRIM folder)."""
        d_ = sr["srimdir"].get().strip()
        out = sr["outdir"].get().strip()
        if not os.path.isdir(d_):
            messagebox.showerror("Reload", f"SRIM folder '{d_}' not found.")
            return
        if out and not os.path.isdir(out):
            messagebox.showerror("Reload", f"Results folder '{out}' not "
                                           f"found.")
            return
        if hooks.get("srim_busy"):
            messagebox.showinfo("Reload", "TRIM is still running.")
            return
        if out:
            try:
                n_ = len(srim_copy_outputs(d_, out, srim_prefix()))
            except Exception as e:
                messagebox.showerror("Reload", f"Copy failed:\n{e}")
                return
            lbl_sres.config(text=f"{n_} file(s) copied from {d_} to {out}.")
            v_sres.set(out)
        else:
            v_sres.set(srim_output_folder(d_))
            lbl_sres.config(text="No results folder (Structure › TRIM "
                                 "run): files of the SRIM folder.")
        sres_cache.clear()
        sres_scan()

    tv_sr = ttk.Treeview(tab_sr, columns=("file", "desc"), show="headings",
                         height=8, selectmode="extended")
    tv_sr.heading("file", text="File")
    tv_sr.heading("desc", text="Content")
    tv_sr.column("file", width=105, stretch=False)
    tv_sr.column("desc", width=300)
    tv_sr.tag_configure("text", foreground="#888888")
    tv_sr.pack(fill="x", pady=(6, 0))
    ttk.Label(tab_sr, text="Select one or more files (Ctrl/Shift), then "
                           "the curves to plot.", foreground="#555555").pack(
        anchor="w", pady=(2, 0))

    lf_cur = ttk.LabelFrame(tab_sr, text="Curves", padding=6)
    lf_cur.pack(fill="x", pady=(6, 0))
    lb_cur = tk.Listbox(lf_cur, selectmode="extended", height=7,
                        exportselection=False)
    lb_cur.pack(fill="x")
    fr_sren = ttk.Frame(tab_sr)
    fr_sren.pack(fill="x", pady=(4, 0))
    ttk.Button(fr_sren, text="Rename & export…",
               command=lambda: sres_rename_export()).pack(side="left")
    ttk.Label(fr_sren, text="copies the selected files with a name "
                            "prefix (e.g. He100keV_RANGE.txt)",
              foreground="#555555").pack(side="left", padx=6)

    def sres_rename_export():
        """Copy the files selected in the file list, with a prefix typed
        by the user before the original name, to a folder (the same by
        default). Asks before overwriting."""
        import shutil
        from tkinter import simpledialog
        src_d = v_sres.get().strip()
        sel_ = list(tv_sr.selection())
        if not sel_:
            messagebox.showinfo("Rename & export", "Select the files in the "
                                                   "list first (Ctrl/Shift-"
                                                   "click selects several).")
            return
        pre = simpledialog.askstring(
            "Rename & export", f"Prefix for the names of the {len(sel_)} "
                               f"file(s)\n(e.g. 'He100keV' → "
                               f"He100keV_{sel_[0]}):", parent=root)
        if pre is None:
            return
        pre = pre.strip()
        bad = set('\\/:*?"<>|')
        if not pre or any(ch in bad for ch in pre):
            messagebox.showerror("Rename & export", "Type a prefix without "
                                                    "\\ / : * ? \" < > |.")
            return
        if pre[-1] not in "_-. ":
            pre += "_"
        dst_d = filedialog.askdirectory(title="Export the renamed files to",
                                        initialdir=src_d or None)
        if not dst_d:
            return
        names = [(f_, pre + f_) for f_ in sel_]
        exist = [n_ for _, n_ in names
                 if os.path.exists(os.path.join(dst_d, n_))]
        if exist and not messagebox.askyesno(
                "Rename & export", "These files already exist and will be "
                                   "replaced:\n" + "\n".join(exist) +
                                   "\n\nContinue?"):
            return
        done, errs = [], []
        for f_, n_ in names:
            try:
                shutil.copy2(os.path.join(src_d, f_), os.path.join(dst_d, n_))
                done.append(n_)
            except Exception as e:
                errs.append(f"{f_}: {e}")
        if errs:
            messagebox.showerror("Rename & export", "Not copied:\n" +
                                 "\n".join(errs))
        lbl_sres.config(text=f"{len(done)} file(s) exported to {dst_d}: "
                             f"{', '.join(done)}")

    fr_so = ttk.Frame(tab_sr)
    fr_so.pack(fill="x", pady=(6, 0))
    ttk.Label(fr_so, text="Depth in").pack(side="left")
    v_sunit = tk.StringVar(value="Å")
    ttk.Combobox(fr_so, textvariable=v_sunit, values=["Å", "nm", "µm"],
                 state="readonly", width=4).pack(side="left", padx=(4, 10))
    v_slog = tk.BooleanVar(value=False)
    v_snorm = tk.BooleanVar(value=False)
    v_sif = tk.BooleanVar(value=True)
    v_sfill = tk.BooleanVar(value=True)
    v_scur = tk.BooleanVar(value=True)
    for txt, var in (("Log y", v_slog), ("Max = 1", v_snorm),
                     ("Fill", v_sfill), ("Cursor", v_scur),
                     ("Layer interfaces", v_sif)):
        ttk.Checkbutton(fr_so, text=txt, variable=var,
                        command=lambda: sres_plot()).pack(side="left",
                                                          padx=(0, 8))
    ttk.Button(fr_so, text="Export plot (.txt)…",
               command=lambda: sres_export()).pack(side="left")
    v_sunit.trace_add("write", lambda *_: sres_plot())

    ttk.Label(tab_sr, text="File content", font=("Segoe UI", 9,
                                                 "bold")).pack(
        anchor="w", pady=(6, 0))
    txt_sres = ScrolledText(tab_sr, height=10, font=("Courier", 9),
                            wrap="none")
    txt_sres.pack(fill="both", expand=True)

    # right pane of Results: the plot
    res_area = ttk.Frame(right_s)
    fig_s = Figure(figsize=(8, 5), dpi=100)
    ax_s = fig_s.add_subplot(111)
    canvas_s = FigureCanvasTkAgg(fig_s, master=res_area)
    toolbar_s = NavigationToolbar2Tk(canvas_s, res_area)
    toolbar_s.update()
    toolbar_s.pack(side="bottom", fill="x")
    canvas_s.get_tk_widget().pack(fill="both", expand=True)

    sres_cache = {}          # path -> (mtime, curves) read
    sres_curves = []         # [(file, label)] in the curve list
    # cursor: vertical line dragged with the mouse, with x and the y of
    # each curve shown while it moves
    sres_cur = {"x": None, "drag": False, "series": [], "line": None,
                "dots": [], "text": None}
    SUM_LABEL = "Sum (all the curves of the list)"

    def sres_scan():
        """List the known SRIM output files of the folder."""
        d_ = v_sres.get().strip()
        tv_sr.delete(*tv_sr.get_children())
        if not os.path.isdir(d_):
            sres_files_changed()
            return
        # known TRIM files, also with a structure name before them
        # (Au_SiO2_RANGE.txt): grouped by structure, in the usual order
        order = list(SRIM_OUTPUTS)
        found = []
        for f_ in os.listdir(d_):
            key, pre = srim_known(f_)
            if key and os.path.isfile(os.path.join(d_, f_)):
                found.append((pre.upper(), order.index(key), f_, key))
        for _, _, f_, key in sorted(found):
            desc, kind, _, _ = SRIM_OUTPUTS[key]
            tv_sr.insert("", "end", iid=f_, values=(f_, desc),
                         tags=("text",) if kind == "text" else ())
        sres_files_changed()

    def sres_read(f_):
        """Curves of an output file: [(label, x, y, xlabel, ylabel,
        is_depth)]; read once (again if the file changed)."""
        path = os.path.join(v_sres.get().strip(), f_)
        mt = os.path.getmtime(path)
        if path in sres_cache and sres_cache[path][0] == mt:
            return sres_cache[path][1]
        key, pre = srim_known(f_)          # also 'Au_SiO2_RANGE.txt'
        desc, kind, cols, group = SRIM_OUTPUTS.get(
            key, (f_, "text", None, None))
        stem = os.path.splitext(f_)[0]
        curves = []
        if kind == "depth":
            data = srim_table(path)
            atoms, _ = read_trim_in(trim_in_of(os.path.dirname(path), pre))
            names = list(cols)
            extra = data.shape[1] - 1 - len(names)
            if group and extra > 0:
                names += [f"{group} {atoms[k]}" if len(atoms) == extra
                          else f"{group} {k + 1}" for k in range(extra)]
            for k, nm in enumerate(names[:data.shape[1] - 1]):
                curves.append((f"{stem}: {nm}", data[:, 0], data[:, k + 1],
                               "Depth", desc[desc.find(" (") + 2:-1]
                               if " (" in desc else "", True))
        elif kind == "range3d":
            data = srim_table(path, max_rows=10 ** 7)
            hy, hx = np.histogram(data[:, 1], bins=100)
            curves.append((f"{stem}: ions vs depth", (hx[:-1] + hx[1:]) / 2,
                           hy.astype(float), "Depth", "Ions per bin", True))
            if data.shape[1] > 3:
                r_ = np.hypot(data[:, 2], data[:, 3])
                hy, hx = np.histogram(r_, bins=100)
                curves.append((f"{stem}: ions vs lateral radius",
                               (hx[:-1] + hx[1:]) / 2, hy.astype(float),
                               "Lateral radius", "Ions per bin", True))
        elif kind == "energy":
            en = srim_energies(path)
            if len(en):
                hy, hx = np.histogram(en, bins=100)
                curves.append((f"{stem}: energy spectrum ({len(en)})",
                               (hx[:-1] + hx[1:]) / 2, hy.astype(float),
                               "Energy (keV)", "Particles per bin", False))
        sres_cache[path] = (mt, curves)
        return curves

    def sres_files_changed(_e=None):
        """Selected files -> curve list and text of the first one."""
        sel = tv_sr.selection()
        old = {sres_curves[i] for i in lb_cur.curselection()}
        lb_cur.delete(0, "end")
        sres_curves.clear()
        msgs = []
        for f_ in sel:
            try:
                cs = sres_read(f_)
            except Exception as e:
                msgs.append(f"{f_}: {e}")
                continue
            if not sres_curves and cs:         # first line: the sum
                sres_curves.append(("", SUM_LABEL))
                lb_cur.insert("end", SUM_LABEL)
                lb_cur.itemconfig("end", foreground="#1a5fb4")
                if ("", SUM_LABEL) in old:
                    lb_cur.selection_set("end")
            for k, c in enumerate(cs):
                sres_curves.append((f_, c[0]))
                lb_cur.insert("end", c[0])
                if (f_, c[0]) in old or (not old and k == 0):
                    lb_cur.selection_set("end")
        txt_sres.config(state="normal")
        txt_sres.delete("1.0", "end")
        if sel:
            try:
                with open(os.path.join(v_sres.get().strip(), sel[0]), "r",
                          encoding="latin-1") as f:
                    lines = []
                    for ln in f:
                        lines.append(ln.rstrip("\r\n"))
                        if len(lines) >= 400:
                            lines.append("…")
                            break
                txt_sres.insert("1.0", "\n".join(msgs + lines))
            except Exception as e:
                txt_sres.insert("1.0", str(e))
        txt_sres.config(state="disabled")
        sres_plot()
    tv_sr.bind("<<TreeviewSelect>>", sres_files_changed)
    lb_cur.bind("<<ListboxSelect>>", lambda e: sres_plot())

    def sres_curve(f_, lab):
        for c in sres_read(f_):
            if c[0] == lab:
                return c
        return None

    def sres_sum():
        """Sum of all the curves of the list: on their common x grid (if
        they have different grids, on the union of the x values, each
        curve interpolated and 0 outside its range). ValueError if the
        curves are of depth and of energy."""
        cs = [sres_curve(f_, lab) for f_, lab in sres_curves if f_]
        cs = [c for c in cs if c is not None]
        if not cs:
            raise ValueError("no curves")
        if len({c[5] for c in cs}) > 1:
            raise ValueError("depth and energy curves cannot be summed")
        x0 = cs[0][1]
        if all(len(c[1]) == len(x0) and np.allclose(c[1], x0) for c in cs):
            x = np.asarray(x0, float)
            y = np.sum([c[2] for c in cs], axis=0)
        else:
            x = np.unique(np.concatenate([c[1] for c in cs]))
            y = np.sum([np.interp(x, c[1], c[2], left=0.0, right=0.0)
                        for c in cs], axis=0)
        yls = sorted({c[4] for c in cs})
        return (SUM_LABEL, x, y, cs[0][3], " / ".join(yls), cs[0][5])

    def sres_series():
        """Curves selected, as plotted: [(label, x, y, xlabel, ylabel,
        is_depth)] with the depth unit and 'Max = 1' applied; and the
        messages of what could not be done."""
        unit = v_sunit.get()
        div = {"Å": 1.0, "nm": 10.0, "µm": 1e4}[unit]
        out, msgs = [], []
        for i in lb_cur.curselection():
            f_, lab = sres_curves[i]
            if not f_:
                try:
                    c = sres_sum()
                except ValueError as e:
                    msgs.append(f"Sum: {e}")
                    continue
            else:
                c = sres_curve(f_, lab)
                if c is None:
                    continue
            lab, x, y, xl, yl, is_depth = c
            if is_depth:
                x = x / div
                xl = f"{xl} ({unit})"
            if v_snorm.get() and np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
                yl = "Max = 1"
            out.append((lab, x, y, xl, yl, is_depth))
        return out, msgs

    def sres_plot():
        ax_s.clear()
        unit = v_sunit.get()
        div = {"Å": 1.0, "nm": 10.0, "µm": 1e4}[unit]
        xlabs, ylabs, depth_any = set(), set(), False
        series, msgs = sres_series()
        # filled as in TRIM: the curves with the largest area behind, so
        # that the smaller ones stay visible in front of them
        area = [float(np.sum(np.abs(s[2]))) for s in series]
        rank = {k: r for r, k in enumerate(
            sorted(range(len(series)), key=lambda k: -area[k]))}
        pos = [y for s in series for y in np.asarray(s[2]) if y > 0]
        base = min(pos) / 2 if v_slog.get() and pos else 0.0
        for n_, (lab, x, y, xl, yl, is_depth) in enumerate(series):
            is_sum = lab == SUM_LABEL
            col = "#000000" if is_sum else PALETTE[n_ % len(PALETTE)]
            z = 2 + 2 * rank[n_]
            if v_sfill.get():
                ax_s.fill_between(x, np.maximum(y, base), base, step="mid",
                                  color="#bbbbbb" if is_sum else col,
                                  alpha=0.9, lw=0, zorder=z)
            ax_s.plot(x, y, color=col, label=lab, drawstyle="steps-mid",
                      lw=2 if is_sum else 1.5, zorder=z + 1)
            xlabs.add(xl)
            ylabs.add(yl)
            depth_any = depth_any or is_depth
        if msgs:
            ax_s.text(0.01, 0.01, "\n".join(msgs), transform=ax_s.transAxes,
                      ha="left", va="bottom", color="#c62828", fontsize=8,
                      zorder=2003)
        if not xlabs:
            ax_s.text(0.5, 0.5, "Select SRIM output files and curves",
                      transform=ax_s.transAxes, ha="center",
                      color="#888888")
        else:
            ax_s.set_xlabel(" / ".join(sorted(xlabs)))
            ax_s.set_ylabel("Max = 1" if v_snorm.get()
                            else " / ".join(sorted(ylabs)))
            if v_slog.get():
                ax_s.set_yscale("log")
            elif v_sfill.get() and all(np.min(s[2]) >= 0 for s in series):
                ax_s.set_ylim(bottom=0)
            if depth_any and v_sif.get():
                # TRIM.IN of the structure of the first curve selected
                pre = next((srim_known(sres_curves[i][0])[1]
                            for i in lb_cur.curselection()
                            if sres_curves[i][0]), "")
                _, widths = read_trim_in(trim_in_of(v_sres.get().strip(),
                                                    pre))
                z = 0.0
                for w_ in widths[:-1]:
                    z += w_
                    ax_s.axvline(z / div, color="#555555", ls="--", lw=1,
                                 zorder=1000)
            ax_s.legend(fontsize=8, loc="upper right")
            ax_s.grid(alpha=0.3)
        sres_cursor_draw(series)
        fig_s.tight_layout()
        canvas_s.draw_idle()

    def sres_cursor_draw(series):
        """Cursor line, dots on the curves and the box with the values
        (kept at the same x when the plot is redrawn, if still inside)."""
        sres_cur.update(series=series, line=None, dots=[], text=None)
        if not v_scur.get() or not series:
            return
        lo = min(float(np.min(s[1])) for s in series)
        hi = max(float(np.max(s[1])) for s in series)
        x = sres_cur["x"]
        if x is None or not lo <= x <= hi:
            x = (lo + hi) / 2
        sres_cur["line"] = ax_s.axvline(x, color="#000000", ls=":", lw=1.2,
                                        zorder=2000)
        for k, s in enumerate(series):
            sres_cur["dots"].append(ax_s.plot(
                [], [], "o", ms=5, zorder=2001, mec="#000000", mew=0.8,
                color=ax_s.lines[k].get_color())[0])
        sres_cur["text"] = ax_s.text(
            0.01, 0.99, "", transform=ax_s.transAxes, ha="left", va="top",
            fontsize=8, family="monospace", zorder=2002,
            bbox=dict(boxstyle="round", fc="#ffffff", ec="#888888",
                      alpha=0.9))
        sres_cursor_set(x)

    def sres_cursor_set(x):
        """Move the cursor to x: the y of each curve is the value of the
        bin (steps) at x."""
        sres_cur["x"] = x
        if sres_cur["line"] is None:
            return
        sres_cur["line"].set_xdata([x, x])
        xl = sres_cur["series"][0][3]
        rows = [f"x = {x:.5g}  ({xl})"]
        for s, dot in zip(sres_cur["series"], sres_cur["dots"]):
            xs, ys = np.asarray(s[1]), np.asarray(s[2])
            if not len(xs) or x < xs.min() or x > xs.max():
                dot.set_data([], [])
                rows.append(f"{s[0]}: –")
                continue
            k = int(np.argmin(np.abs(xs - x)))
            dot.set_data([x], [ys[k]])
            rows.append(f"{s[0]}: {ys[k]:.5g}")
        sres_cur["text"].set_text("\n".join(rows))

    def sres_on_press(e):
        if (e.inaxes is not ax_s or e.button != 1 or sres_cur["line"] is None
                or toolbar_s.mode):
            return
        sres_cur["drag"] = True
        sres_cursor_set(e.xdata)
        canvas_s.draw_idle()

    def sres_on_move(e):
        if not sres_cur["drag"] or e.inaxes is not ax_s or e.xdata is None:
            return
        sres_cursor_set(e.xdata)
        canvas_s.draw_idle()

    def sres_on_release(_e):
        sres_cur["drag"] = False
    canvas_s.mpl_connect("button_press_event", sres_on_press)
    canvas_s.mpl_connect("motion_notify_event", sres_on_move)
    canvas_s.mpl_connect("button_release_event", sres_on_release)

    def sres_export():
        """Save the curves of the plot in a .txt file (name asked):
        one x column and one column per curve if they share the x values,
        else one block per curve."""
        series, msgs = sres_series()
        if not series:
            messagebox.showinfo("Export plot", "No curves in the plot."
                                + ("\n" + "\n".join(msgs) if msgs else ""))
            return
        d_ = v_sres.get().strip()
        path = filedialog.asksaveasfilename(
            title="Export the plotted curves", defaultextension=".txt",
            initialdir=d_ if os.path.isdir(d_) else None,
            initialfile="srim_curves.txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not path:
            return
        head = ["# SRIM results exported by ionolum",
                f"# Folder = {d_}",
                f"# Normalised (Max = 1) = {'yes' if v_snorm.get() else 'no'}"]
        head += [f"# Curve {k + 1} = {s[0]}  [{s[4]}]"
                 for k, s in enumerate(series)]
        x0 = series[0][1]
        same = all(len(s[1]) == len(x0) and np.allclose(s[1], x0)
                   for s in series)
        lines = list(head)
        if same:
            lines.append("# " + "\t".join([series[0][3]] +
                                          [s[0] for s in series]))
            for k in range(len(x0)):
                lines.append("\t".join([f"{x0[k]:.6g}"] +
                                       [f"{s[2][k]:.6g}" for s in series]))
        else:
            for s in series:
                lines += ["", f"# {s[0]}", f"# {s[3]}\t{s[4]}"]
                lines += [f"{a:.6g}\t{b:.6g}" for a, b in zip(s[1], s[2])]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            messagebox.showerror("Export plot", f"Could not write "
                                                f"{path}:\n{e}")
            return
        lbl_sres.config(text=f"{len(series)} curve(s) exported to {path}")

    # ----------------- SRIM sub-tab: Compound dictionary -----------------
    # Compounds of SRIM (Data/Compound.dat of the SRIM folder): density,
    # composition, phase; 'Add to Target layers' adds one as a SRIM layer
    # of Structure (srim_extra / srim_ovr / srim_rho / srim_gas).
    tab_cd = ttk.Frame(sub_s, padding=8)
    sub_s.add(tab_cd, text="  Compound dictionary  ")
    cdd = {"all": [], "shown": []}
    v_cdfile = tk.StringVar(value=os.path.join(
        sr["srimdir"].get().strip() or r"C:\Program Files (x86)\SRIM",
        "Data", "Compound.dat"))
    fr_cd0 = ttk.Frame(tab_cd)
    fr_cd0.pack(fill="x")
    ttk.Label(fr_cd0, text="File:", font=("Segoe UI", 9, "bold")).pack(
        side="left")
    tk.Entry(fr_cd0, textvariable=v_cdfile).pack(side="left", padx=4,
                                                 fill="x", expand=True)
    def cd_browse():
        p_ = filedialog.askopenfilename(
            title="SRIM Compound.dat",
            initialdir=os.path.dirname(v_cdfile.get()) or None,
            filetypes=[("SRIM compounds", "*.dat"), ("All", "*.*")])
        if p_:
            v_cdfile.set(os.path.normpath(p_))
            cd_load()
    ttk.Button(fr_cd0, text="Compound.dat…", command=cd_browse).pack(
        side="left")
    ttk.Button(fr_cd0, text="Reload", command=lambda: cd_load()).pack(
        side="left", padx=(4, 0))
    fr_cd1 = ttk.Frame(tab_cd)
    fr_cd1.pack(fill="x", pady=(6, 0))
    ttk.Label(fr_cd1, text="Table").pack(side="left")
    v_cdcat = tk.StringVar(value="All")
    cb_cdcat = ttk.Combobox(fr_cd1, textvariable=v_cdcat, state="readonly",
                            width=26)
    cb_cdcat.pack(side="left", padx=4)
    ttk.Label(fr_cd1, text="Search").pack(side="left", padx=(8, 0))
    v_cdfind = tk.StringVar(value="")
    tk.Entry(fr_cd1, textvariable=v_cdfind, width=18).pack(side="left",
                                                           padx=4)
    fr_cdl = ttk.Frame(tab_cd)
    fr_cdl.pack(fill="both", expand=True, pady=(6, 0))
    tv_cd = ttk.Treeview(fr_cdl, columns=("name", "rho", "comp"),
                         show="headings", height=12, selectmode="browse")
    for c, t_, w_ in (("name", "Compound", 190), ("rho", "ρ (g/cm³)", 75),
                      ("comp", "Composition (atom fractions)", 200)):
        tv_cd.heading(c, text=t_)
        tv_cd.column(c, width=w_, anchor="e" if c == "rho" else "w",
                     stretch=c != "rho")
    sb_cd = ttk.Scrollbar(fr_cdl, orient="vertical", command=tv_cd.yview)
    tv_cd.configure(yscrollcommand=sb_cd.set)
    sb_cd.pack(side="right", fill="y")
    tv_cd.pack(side="left", fill="both", expand=True)
    lf_cdd = ttk.LabelFrame(tab_cd, text="Compound", padding=6)
    lf_cdd.pack(fill="x", pady=(6, 0))
    lbl_cdd = ttk.Label(lf_cdd, text="", wraplength=420, justify="left")
    lbl_cdd.pack(anchor="w")
    tv_cde = ttk.Treeview(lf_cdd, columns=("z", "el", "given", "frac"),
                          show="headings", height=4)
    for c, t_, w_ in (("z", "Z", 40), ("el", "Element", 70),
                      ("given", "In the file", 110),
                      ("frac", "Atom fraction", 110)):
        tv_cde.heading(c, text=t_)
        tv_cde.column(c, width=w_, anchor="e" if c != "el" else "center")
    tv_cde.pack(fill="x", pady=(4, 0))
    lbl_cdc = ttk.Label(lf_cdd, text="", wraplength=420, justify="left",
                        foreground="#555555", font=("Segoe UI", 8))
    lbl_cdc.pack(anchor="w", pady=(4, 0))
    lf_cda = ttk.LabelFrame(tab_cd, text="Add to Structure › Target layers",
                            padding=6)
    lf_cda.pack(fill="x", pady=(6, 0))
    ttk.Label(lf_cda, text="Thickness").grid(row=0, column=0, sticky="w")
    v_cdthick = tk.StringVar(value="1000")
    ttk.Entry(lf_cda, textvariable=v_cdthick, width=10,
              justify="right").grid(row=0, column=1, sticky="w", padx=4)
    v_cdunit = tk.StringVar(value="Å")
    ttk.Combobox(lf_cda, textvariable=v_cdunit, state="readonly", width=12,
                 values=["Å", "nm", "µm", "10¹⁵ at/cm²"]).grid(
        row=0, column=2, sticky="w")
    ttk.Label(lf_cda, text="ρ (g/cm³)").grid(row=0, column=3, sticky="w",
                                             padx=(10, 2))
    v_cdrho = tk.StringVar(value="")
    ttk.Entry(lf_cda, textvariable=v_cdrho, width=10,
              justify="right").grid(row=0, column=4, sticky="w")
    v_cdgas = tk.BooleanVar(value=False)
    ttk.Checkbutton(lf_cda, text="Gas", variable=v_cdgas).grid(
        row=0, column=5, sticky="w", padx=(10, 0))
    ttk.Button(lf_cda, text="Add to Target layers",
               command=lambda: cd_add()).grid(row=1, column=0, columnspan=3,
                                              sticky="w", pady=(6, 0))
    lbl_cda = ttk.Label(lf_cda, text="", foreground="#555555",
                        wraplength=420)
    lbl_cda.grid(row=2, column=0, columnspan=6, sticky="w", pady=(4, 0))

    def cd_comp_text(c):
        return " ".join("%s %.3g" % (s_, f_) for s_, f_ in c["comp"].items())

    def cd_load():
        p_ = v_cdfile.get().strip()
        try:
            cdd["all"] = read_srim_compounds(p_)
        except OSError as e:
            cdd["all"] = []
            lbl_cdd.config(text=f"Cannot read {p_}:\n{e}",
                           foreground="#c62828")
            cd_filter()
            return
        cats = []
        for c in cdd["all"]:
            if c["category"] not in cats:
                cats.append(c["category"])
        cb_cdcat.configure(values=["All"] + cats)
        if v_cdcat.get() not in ["All"] + cats:
            v_cdcat.set("All")
        lbl_cdd.config(text=f"{len(cdd['all'])} compounds in {p_}.",
                       foreground="#555555")
        cd_filter()

    def cd_filter(*_):
        cat, find = v_cdcat.get(), v_cdfind.get().strip().lower()
        cdd["shown"] = [
            c for c in cdd["all"]
            if (cat == "All" or c["category"] == cat) and
            (not find or find in c["name"].lower() or
             find in c["description"].lower() or
             find in cd_comp_text(c).lower())]
        tv_cd.delete(*tv_cd.get_children())
        for k, c in enumerate(cdd["shown"]):
            tv_cd.insert("", "end", iid=str(k), values=(
                c["name"], f"{c['density']:.5g}" + (" (gas)" if c["gas"]
                                                    else ""),
                cd_comp_text(c)))

    v_cdcat.trace_add("write", cd_filter)
    v_cdfind.trace_add("write", cd_filter)

    def cd_current():
        sel_ = tv_cd.selection()
        return cdd["shown"][int(sel_[0])] if sel_ else None

    def cd_show(_e=None):
        c = cd_current()
        tv_cde.delete(*tv_cde.get_children())
        if c is None:
            return
        n_, rho, m_ = layer_density(list(c["comp"].items()))
        n_at = c["density"] * N_AVOGADRO / m_ if m_ else None
        lbl_cdd.config(text=(
            f"{c['name']}\n{c['description']}\n"
            f"Table: {c['category']}\n"
            f"Density ρ = {c['density']:.6g} g/cm³"
            + (f"  →  n = {n_at:.4e} atoms/cm³" if n_at else "")
            + ("   (gas)" if c["gas"] else "")
            + f"\nComposition given in "
              f"{'mass %' if c['mass_percent'] else 'atom % / atoms'}"),
            foreground="#000000")
        for z, s_, v in c["given"]:
            tv_cde.insert("", "end", values=(
                z, s_, f"{v:.5g}", f"{c['comp'].get(s_, 0):.5f}"))
        tv_cde.config(height=max(2, min(8, len(c["given"]))))
        lbl_cdc.config(text="\n".join(c["comments"][:6]))
        v_cdrho.set(f"{c['density']:.6g}")
        v_cdgas.set(c["gas"])

    tv_cd.bind("<<TreeviewSelect>>", cd_show)

    def cd_add():
        """The selected compound -> a new SRIM layer (Structure › Target
        layers), after the selected layer of that list or at the end."""
        c = cd_current()
        if c is None:
            lbl_cda.config(text="Select a compound in the list.",
                           foreground="#c62828")
            return
        try:
            t_ = float(v_cdthick.get().replace(",", "."))
            rho = float(v_cdrho.get().replace(",", "."))
            if t_ <= 0 or rho <= 0:
                raise ValueError
        except ValueError:
            lbl_cda.config(text="Thickness and ρ must be numbers > 0.",
                           foreground="#c62828")
            return
        n_ = 1
        while f"x{n_}" in srim_extra:
            n_ += 1
        k_ = f"x{n_}"
        unit = v_cdunit.get()
        ov = {"name": c["name"], "comp": dict(c["comp"])}
        if unit.startswith("10"):
            ov["thick"] = t_                   # 10¹⁵ at/cm²
        else:
            ov["width"] = t_ * {"Å": 1.0, "nm": 10.0, "µm": 1e4}[unit]
        srim_extra[k_] = {}
        srim_ovr[k_] = ov
        srim_rho[k_] = rho
        srim_gas[k_] = bool(v_cdgas.get())
        sel_ = [srim_key(s_) for s_ in tv_sl.selection()]
        pos = [j for j, k in enumerate(srim_order) if k in sel_]
        srim_order.insert(pos[-1] + 1 if pos else len(srim_order), k_)
        srim_refresh()
        tv_sl.selection_set(k_)
        tv_sl.see(k_)
        lbl_cda.config(text=f"Added '{c['name']}' ({t_:g} {unit}, ρ = "
                            f"{rho:g} g/cm³{', gas' if v_cdgas.get() else ''})"
                            f" to Structure › Target layers.",
                       foreground="#2e7d32")

    cd_load()

    # ------------------- SRIM sub-tab: Stopping & Range ----------------------
    # Stopping and range tables of SRIM's SR Module (SRModule.exe in the
    # 'SR Module' folder of SRIM): SR.IN is written there, the program
    # writes the table in ionolum_SR.txt, which is read and plotted.
    tab_sp = ttk.Frame(sub_s, padding=8)
    sub_s.add(tab_sp, text="  Stopping & Range  ")
    v_spdir = tk.StringVar(value=r"C:\Program Files (x86)\SRIM\SR Module")
    fr_sp0 = ttk.Frame(tab_sp)
    fr_sp0.pack(fill="x")
    ttk.Label(fr_sp0, text="SR Module folder:",
              font=("Segoe UI", 9, "bold")).pack(side="left")
    tk.Entry(fr_sp0, textvariable=v_spdir).pack(side="left", padx=4,
                                                 fill="x", expand=True)
    ttk.Button(fr_sp0, text="SR Module folder…", command=lambda: (
        lambda d_: d_ and v_spdir.set(os.path.normpath(d_)))(
        filedialog.askdirectory(title="SR Module folder of SRIM (with "
                                      "SRModule.exe)",
                                initialdir=v_spdir.get() or None))).pack(
        side="left")
    lbl_spion = ttk.Label(tab_sp, text="", foreground="#555555")
    lbl_spion.pack(anchor="w", pady=(6, 0))

    lf_spt = ttk.LabelFrame(tab_sp, text="Target", padding=6)
    lf_spt.pack(fill="x", pady=(6, 0))
    lf_spt.columnconfigure(1, weight=1)
    v_spsrc = tk.StringVar(value="layer")
    ttk.Radiobutton(lf_spt, text="Layer of Structure ›", value="layer",
                    variable=v_spsrc).grid(row=0, column=0, sticky="w")
    v_splayer = tk.StringVar(value="")
    cb_splayer = ttk.Combobox(lf_spt, textvariable=v_splayer,
                              state="readonly")
    cb_splayer.grid(row=0, column=1, columnspan=2, sticky="we", padx=4)
    ttk.Radiobutton(lf_spt, text="Composition", value="comp",
                    variable=v_spsrc).grid(row=1, column=0, sticky="w")
    v_spcomp = tk.StringVar(value="")
    tk.Entry(lf_spt, textvariable=v_spcomp).grid(row=1, column=1,
                                                  columnspan=2, sticky="we",
                                                  padx=4, pady=1)
    ttk.Label(lf_spt, text="ρ (g/cm³)").grid(row=2, column=0, sticky="e")
    v_sprho = tk.StringVar(value="")
    tk.Entry(lf_spt, textvariable=v_sprho, width=10, justify="right").grid(
        row=2, column=1, sticky="w", padx=4, pady=1)
    v_spgas = tk.BooleanVar(value=False)
    ttk.Checkbutton(lf_spt, text="Gas", variable=v_spgas).grid(
        row=2, column=2, sticky="w")
    ttk.Label(lf_spt, text="Composition 'Si 1 O 2' or 'SiO2'; with a layer, "
                           "its composition and ρ are used.",
              foreground="#555555", wraplength=400).grid(
        row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))

    lf_spo = ttk.LabelFrame(tab_sp, text="Calculation", padding=6)
    lf_spo.pack(fill="x", pady=(6, 0))
    v_spemin = tk.StringVar(value="10")
    v_spemax = tk.StringVar(value="10000")
    v_spcorr = tk.StringVar(value="1")
    v_spunits = tk.StringVar(value=SR_UNITS[6])
    for r_, (txt, var) in enumerate((("E min (keV)", v_spemin),
                                     ("E max (keV)", v_spemax),
                                     ("Compound correction", v_spcorr))):
        ttk.Label(lf_spo, text=txt).grid(row=r_, column=0, sticky="w",
                                         pady=1)
        tk.Entry(lf_spo, textvariable=var, width=10, justify="right").grid(
            row=r_, column=1, sticky="w", padx=4, pady=1)
    ttk.Label(lf_spo, text="Stopping units").grid(row=3, column=0,
                                                  sticky="w", pady=1)
    ttk.Combobox(lf_spo, textvariable=v_spunits, values=SR_UNITS,
                 state="readonly", width=28).grid(row=3, column=1,
                                                  sticky="w", padx=4)
    fr_spb = ttk.Frame(tab_sp)
    fr_spb.pack(fill="x", pady=(6, 0))
    ttk.Button(fr_spb, text="Run SR", command=lambda: sp_run()).pack(
        side="left")
    b_spstop = ttk.Button(fr_spb, text="Stop SR", state="disabled",
                          command=lambda: sp_stop())
    b_spstop.pack(side="left", padx=4)
    ttk.Button(fr_spb, text="Load SR output…",
               command=lambda: sp_load()).pack(side="left", padx=4)
    ttk.Button(fr_spb, text="Export table (.txt)…",
               command=lambda: sp_export()).pack(side="left")
    lbl_sp = ttk.Label(tab_sp, text="", foreground="#555555",
                       wraplength=420)
    lbl_sp.pack(anchor="w", pady=(4, 0))
    tv_sp = ttk.Treeview(tab_sp, columns=("ly", "e", "el", "nu", "rg", "lo",
                                          "la"),
                         show="headings", height=12)
    tv_sp.heading("ly", text="Target")
    tv_sp.column("ly", width=110, anchor="w")
    for c_, t_, w_ in (("e", "E (keV)", 70), ("el", "dE/dx elec.", 80),
                       ("nu", "dE/dx nucl.", 80), ("rg", "Range (Å)", 80),
                       ("lo", "Long. str. (Å)", 85),
                       ("la", "Lat. str. (Å)", 80)):
        tv_sp.heading(c_, text=t_)
        tv_sp.column(c_, width=w_, anchor="e")
    tv_sp.pack(fill="both", expand=True, pady=(4, 0))
    fr_spo2 = ttk.Frame(tab_sp)
    fr_spo2.pack(fill="x", pady=(4, 0))
    v_spshow = {k: tk.BooleanVar(value=v) for k, v in (
        ("elec", True), ("nucl", True), ("tot", True), ("range", True))}
    for k_, txt in (("elec", "Electronic"), ("nucl", "Nuclear"),
                    ("tot", "Total"), ("range", "Range (right axis)")):
        ttk.Checkbutton(fr_spo2, text=txt, variable=v_spshow[k_],
                        command=lambda: sp_plot()).pack(side="left",
                                                        padx=(0, 6))
    fr_spo3 = ttk.Frame(tab_sp)
    fr_spo3.pack(fill="x", pady=(4, 0))
    ttk.Label(fr_spo3, text="Range in").pack(side="left")
    v_sprunit = tk.StringVar(value="Å")
    cb_sprunit = ttk.Combobox(fr_spo3, textvariable=v_sprunit,
                              values=["Å", "nm", "µm"], state="readonly",
                              width=4)
    cb_sprunit.pack(side="left", padx=(4, 10))
    cb_sprunit.bind("<<ComboboxSelected>>", lambda e: sp_redisplay())
    v_splogx = tk.BooleanVar(value=True)
    ttk.Checkbutton(fr_spo3, text="Log x", variable=v_splogx,
                    command=lambda: (sp_plot(), sp_bragg())).pack(
        side="left", padx=(0, 10))
    v_spcur = tk.BooleanVar(value=True)
    ttk.Checkbutton(fr_spo3, text="Cursor", variable=v_spcur,
                    command=lambda: (sp_plot(), sp_bragg())).pack(
        side="left")
    SP_RDIV = {"Å": 1.0, "nm": 10.0, "µm": 1e4}

    # right pane: tab 'Stopping & range' (vs energy) and tab 'Bragg peak'
    # (total stopping vs depth along the ion path)
    sp_area = ttk.Frame(right_s)
    nb_sp = ttk.Notebook(sp_area)
    nb_sp.pack(fill="both", expand=True)
    fr_sp1 = ttk.Frame(nb_sp)
    fr_sp2 = ttk.Frame(nb_sp)
    nb_sp.add(fr_sp1, text="  Stopping & range  ")
    nb_sp.add(fr_sp2, text="  Bragg peak  ")
    fig_sp = Figure(figsize=(8, 5), dpi=100)
    ax_sp = fig_sp.add_subplot(111)
    ax_sp2 = ax_sp.twinx()
    canvas_sp = FigureCanvasTkAgg(fig_sp, master=fr_sp1)
    toolbar_sp = NavigationToolbar2Tk(canvas_sp, fr_sp1)
    toolbar_sp.update()
    toolbar_sp.pack(side="bottom", fill="x")
    canvas_sp.get_tk_widget().pack(fill="both", expand=True)
    # Bragg peak tab: initial energy, plot
    fr_bgo = ttk.Frame(fr_sp2, padding=(6, 4))
    fr_bgo.pack(side="top", fill="x")
    ttk.Label(fr_bgo, text="Initial energy E₀ (keV)").pack(side="left")
    v_bge0 = tk.StringVar(value="")
    e_bge0 = ttk.Entry(fr_bgo, textvariable=v_bge0, width=9,
                       justify="right")
    e_bge0.pack(side="left", padx=4)
    e_bge0.bind("<Return>", lambda e: sp_bragg())
    e_bge0.bind("<FocusOut>", lambda e: sp_bragg())
    ttk.Button(fr_bgo, text="Ion energy of Structure",
               command=lambda: (v_bge0.set(sr["energy"].get().strip()),
                                sp_bragg())).pack(side="left", padx=4)
    lbl_bg = ttk.Label(fr_bgo, text="", foreground="#555555")
    lbl_bg.pack(side="left", padx=6)
    fig_bg = Figure(figsize=(8, 5), dpi=100)
    ax_bg = fig_bg.add_subplot(111)
    canvas_bg = FigureCanvasTkAgg(fig_bg, master=fr_sp2)
    toolbar_bg = NavigationToolbar2Tk(canvas_bg, fr_sp2)
    toolbar_bg.update()
    toolbar_bg.pack(side="bottom", fill="x")
    canvas_bg.get_tk_widget().pack(fill="both", expand=True)
    sp_data = {"tables": [], "units": "", "unit_no": 7}  # [(title, table)]
    sp_state = [None]

    SP_ALL = "All layers (one table per layer)"

    def sp_refresh():
        """Ion of Structure and the target layers of Structure (with 'All
        layers' when there are several) in the combobox."""
        sym = sr["ion"].get().strip()
        isos = isotope_masses(sym)
        mass = sr["mass"].get().strip() or (f"{isos[0][1]:.4f}" if isos
                                            else "?")
        lbl_spion.config(text=f"Ion (from Structure): {sym} {mass} amu")
        names = [f"{k + 1}: {L['name']} "
                 f"({' '.join(f'{s_}{c:.3g}' for s_, c in L['els'].items())}"
                 f", ρ {L['rho']:.4g})" if L["rho"] else
                 f"{k + 1}: {L['name']} (ρ unknown)"
                 for k, L in enumerate(srim_lays)]
        if len(names) > 1:
            names.append(SP_ALL)
        cb_splayer.config(values=names)
        if names and v_splayer.get() not in names:
            v_splayer.set(names[0])
        if not names:
            v_splayer.set("")

    def sp_jobs():
        """[(SR.IN text, title, output file name)] from the boxes: one for
        a layer or a composition, one per layer for 'All layers'.
        ValueError if something is wrong."""
        import periodictable
        sym = sr["ion"].get().strip()
        sym = sym[:1].upper() + sym[1:].lower()
        isos = isotope_masses(sym)
        if not isos:
            raise ValueError(f"Ion '{sym}' is not an element (Structure).")
        mass = float(sr["mass"].get().strip() or isos[0][1])
        z_ion = periodictable.elements.symbol(sym).number
        try:
            e1 = float(v_spemin.get().replace(",", "."))
            e2 = float(v_spemax.get().replace(",", "."))
            corr = float(v_spcorr.get().replace(",", "."))
        except ValueError:
            raise ValueError("E min, E max and the compound correction "
                             "must be numbers.")
        if not 0 < e1 < e2:
            raise ValueError("0 < E min < E max is needed.")
        units = int(v_spunits.get()[0])
        targets = []                       # (comp, rho, gas, name)
        if v_spsrc.get() == "layer":
            names = list(cb_splayer.cget("values"))
            if v_splayer.get() not in names:
                raise ValueError("No layer: define the target layers in "
                                 "Structure (or type a composition).")
            idx = (range(len(srim_lays)) if v_splayer.get() == SP_ALL
                   else [names.index(v_splayer.get())])
            for k in idx:
                L = srim_lays[k]
                if not L["els"] or not L["rho"]:
                    raise ValueError(f"Layer {k + 1} '{L['name']}': "
                                     f"composition or density unknown.")
                targets.append((L["els"], L["rho"], L["gas"],
                                f"{k + 1}: {L['name']}"))
        else:
            if not v_spcomp.get().strip():
                raise ValueError("Type a composition (e.g. 'SiO2').")
            comp = srim_parse_comp(v_spcomp.get())
            try:
                rho = float(v_sprho.get().replace(",", "."))
            except ValueError:
                raise ValueError("ρ is not a number.")
            targets.append((comp, rho, v_spgas.get(),
                            v_spcomp.get().strip()))
        jobs = []
        for j, (comp, rho, gas, tname) in enumerate(targets, start=1):
            if rho <= 0:
                raise ValueError(f"{tname}: ρ must be > 0.")
            tot = sum(comp.values())
            els = []
            for s_, c in comp.items():
                el = periodictable.elements.symbol(s_)
                els.append((el.number, el.name.capitalize(), c / tot,
                            el.mass))
            out = SR_OUT_NAME if len(targets) == 1 else \
                f"ionolum_SR_{j}.txt"
            jobs.append((sr_in_text(z_ion, mass, int(gas), rho, corr, els,
                                    units, e1, e2, out_name=out),
                         f"{sym} in {tname}", out))
        return jobs

    def sp_out_paths(d_, name=SR_OUT_NAME):
        """Where SRModule.exe may have written an output file: the SR
        Module folder or, for Program Files without rights, the
        VirtualStore."""
        ps = [os.path.join(d_, name)]
        drv, rest = os.path.splitdrive(os.path.abspath(d_))
        if rest.lower().startswith("\\program files"):
            ps.append(os.path.join(os.environ.get("LOCALAPPDATA", ""),
                                   "VirtualStore", rest.lstrip("\\"), name))
        return ps

    def sp_run():
        """For each target: SR.IN in the SR Module folder and SRModule.exe
        (all in one elevated command if the folder cannot be written, as
        for TRIM); then the tables are read and shown."""
        import threading
        import subprocess
        import time
        if sp_state[0]:
            messagebox.showinfo("Run SR", "SR Module is still running.")
            return
        d_ = v_spdir.get().strip()
        exe = os.path.join(d_, "SRModule.exe")
        if not os.path.isfile(exe):
            messagebox.showerror("Run SR", f"SRModule.exe not found in "
                                           f"'{d_}'.")
            return
        try:
            jobs = sp_jobs()
        except Exception as e:
            messagebox.showerror("Run SR", str(e))
            return
        t0 = time.time()
        stage = None
        try:                               # can the folder be written?
            with open(os.path.join(d_, "SR.IN"), "w", encoding="latin-1",
                      newline="\r\n") as f:
                f.write(jobs[0][0])
        except PermissionError:
            import tempfile
            stage = tempfile.mkdtemp(prefix="ibist_sr_")
            for j, (text, _, _) in enumerate(jobs, start=1):
                with open(os.path.join(stage, f"SR_{j}.IN"), "w",
                          encoding="latin-1", newline="\r\n") as f:
                    f.write(text)
        except Exception as e:
            messagebox.showerror("Run SR", str(e))
            return
        state = {"error": None, "admin": False, "proc": None,
                 "handle": None, "jobs": jobs, "stopped": False}
        sp_state[0] = state

        def set_handle(h):
            state["handle"] = h

        def work():
            try:
                if stage:
                    state["admin"] = True
                    parts = [f'cd /d "{d_}"']
                    for j in range(1, len(jobs) + 1):
                        parts += [f'copy /y "{os.path.join(stage, f"SR_{j}.IN")}"'
                                  f' "{os.path.join(d_, "SR.IN")}" >nul',
                                  "SRModule.exe"]
                    run_elevated("cmd.exe", "/c " + " & ".join(parts), d_,
                                 show=7, on_start=set_handle)
                else:
                    for j, (text, _, _) in enumerate(jobs, start=1):
                        if state["stopped"]:
                            break
                        state["cur"] = j
                        with open(os.path.join(d_, "SR.IN"), "w",
                                  encoding="latin-1", newline="\r\n") as f:
                            f.write(text)
                        try:
                            state["proc"] = subprocess.Popen([exe], cwd=d_)
                            state["proc"].wait()
                        except OSError as e:
                            if getattr(e, "winerror", None) != 740:
                                raise
                            state["admin"] = True
                            run_elevated(exe, "", d_, on_start=set_handle)
            except Exception as e:
                state["error"] = e
            state["handle"] = None
            state["done"] = True

        def poll():
            if state is not sp_state[0]:
                return
            if not state.get("done"):
                if len(jobs) > 1 and state.get("cur"):
                    lbl_sp.config(text=f"SR Module running… target "
                                       f"{state['cur']} of {len(jobs)}",
                                  foreground="#1a5fb4")
                root.after(300, poll)
                return
            sp_state[0] = None
            b_spstop.state(["disabled"])
            if state["error"]:
                e_ = state["error"]
                lbl_sp.config(text="SR failed: " + (
                    "administrator rights refused" if getattr(
                        e_, "winerror", None) == 1223 else str(e_)),
                    foreground="#c62828")
                return
            tables, missing = [], []
            for _, title, out in jobs:
                new = [p_ for p_ in sp_out_paths(d_, out)
                       if os.path.isfile(p_) and
                       os.path.getmtime(p_) >= t0 - 2]
                if not new:
                    missing.append(out)
                    continue
                try:
                    tables.append((title, read_sr_output(new[0])))
                except Exception as e:
                    missing.append(f"{out} ({e})")
            if not tables:
                lbl_sp.config(text=f"SR Module ended but no table was "
                                   f"written ({', '.join(missing)}; see "
                                   f"SR.IN in {d_}).", foreground="#c62828")
                return
            sp_show_tables(tables, f"read from {d_}" + (
                f"; not written: {', '.join(missing)}" if missing else ""))
        b_spstop.state(["!disabled"])
        lbl_sp.config(text="SR Module running…", foreground="#1a5fb4")
        threading.Thread(target=work, daemon=True).start()
        root.after(300, poll)

    def sp_stop():
        import subprocess
        state = sp_state[0]
        if state is None:
            return
        state["stopped"] = True
        try:
            if state["proc"] is not None and state["proc"].poll() is None:
                state["proc"].kill()
            elif state["handle"]:
                import ctypes
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                k32.TerminateProcess.argtypes = [ctypes.c_void_p,
                                                 ctypes.c_uint]
                k32.TerminateProcess(state["handle"], 1)
                subprocess.run(["taskkill", "/f", "/t", "/im",
                                "SRModule.exe"], capture_output=True,
                               creationflags=0x08000000)
        except Exception:
            pass
        sp_state[0] = None
        b_spstop.state(["disabled"])
        lbl_sp.config(text="SR Module stopped.", foreground="#c62828")

    def sp_show_tables(tables, where=""):
        """Tables [(title, table)]: in the list (Target column) and in
        the plot."""
        sp_data.update(tables=tables,
                       units=v_spunits.get().split(" - ", 1)[1],
                       unit_no=int(v_spunits.get()[0]))
        lbl_sp.config(text=f"{len(tables)} table(s), "
                           f"{sum(len(t_['energy']) for _, t_ in tables)} "
                           f"rows" + (f"; {where}" if where else ""),
                      foreground="#555555")
        if not v_bge0.get().strip():               # Bragg: ion energy
            v_bge0.set(sr["energy"].get().strip())
        sp_redisplay()

    def sp_redisplay():
        """Table (ranges in the unit chosen), plot and Bragg peak."""
        div = SP_RDIV[v_sprunit.get()]
        u_ = v_sprunit.get()
        for c_, t_ in (("rg", "Range"), ("lo", "Long. str."),
                       ("la", "Lat. str.")):
            tv_sp.heading(c_, text=f"{t_} ({u_})")
        tv_sp.delete(*tv_sp.get_children())
        for title, t_ in sp_data["tables"]:
            for k in range(len(t_["energy"])):
                tv_sp.insert("", "end", values=(title,) + tuple(
                    f"{t_[c][k]:.4g}" for c in ("energy", "elec", "nucl"))
                    + tuple(f"{t_[c][k] / div:.4g}"
                            for c in ("range", "long", "lat")))
        sp_plot()
        sp_bragg()

    def sp_load():
        p_ = filedialog.askopenfilename(
            title="SR Module output file", initialdir=v_spdir.get() or None,
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not p_:
            return
        try:
            t_ = read_sr_output(p_)
        except Exception as e:
            lbl_sp.config(text=f"{p_} not read: {e}", foreground="#c62828")
            return
        sp_show_tables([(os.path.basename(p_), t_)], f"read from {p_}")

    def sp_plot():
        import warnings
        with warnings.catch_warnings():     # clearing a log axis twin
            warnings.simplefilter("ignore", UserWarning)
            ax_sp.clear()
            ax_sp2.clear()
        tables = sp_data["tables"]
        if not tables:
            ax_sp.text(0.5, 0.5, "Run SR (or load an SR output file)",
                       transform=ax_sp.transAxes, ha="center",
                       color="#888888")
            ax_sp2.set_yticks([])
            canvas_sp.draw_idle()
            return
        many = len(tables) > 1
        div = SP_RDIV[v_sprunit.get()]
        kinds = (("elec", "Electronic", "#1a5fb4", "-", 1.4),
                 ("nucl", "Nuclear", "#c62828", ":", 1.4),
                 ("tot", "Total", "#000000", "-", 2.2))
        series = []                    # for the cursor
        for i_, (title, t_) in enumerate(tables):
            e = t_["energy"]
            ys = {"elec": t_["elec"], "nucl": t_["nucl"],
                  "tot": t_["elec"] + t_["nucl"]}
            lc = PALETTE[i_ % len(PALETTE)]
            for k_, lab, col, ls_, lw_ in kinds:
                if v_spshow[k_].get():
                    # one target: a colour per quantity; several: a colour
                    # per target, a line style per quantity
                    lab_ = f"{title} {lab}" if many else lab
                    ax_sp.plot(e, ys[k_], color=lc if many else col,
                               ls=ls_ if many else "-",
                               lw=lw_ if many else 1.8, label=lab_)
                    series.append((lab_, e, ys[k_], lc if many else col,
                                   ax_sp))
            if v_spshow["range"].get():
                lab_ = f"{title} range" if many else "Projected range"
                ax_sp2.plot(e, t_["range"] / div,
                            color=lc if many else "#2e7d32", ls="--",
                            lw=1.4, label=lab_)
                series.append((f"{lab_} ({v_sprunit.get()})", e,
                               t_["range"] / div,
                               lc if many else "#2e7d32", ax_sp2))
        if v_spshow["range"].get():
            ax_sp2.set_ylabel(f"Projected range ({v_sprunit.get()})")
            # clear() puts the twin axis back on the left
            ax_sp2.yaxis.set_label_position("right")
            ax_sp2.yaxis.tick_right()
        else:
            ax_sp2.set_yticks([])
        ax_sp.set_xscale("log" if v_splogx.get() else "linear")
        ax_sp.set_xlabel("Energy (keV)")
        ax_sp.set_ylabel(f"Stopping ({sp_data['units']})")
        ax_sp.set_title(tables[0][0] if not many else
                        f"{len(tables)} targets", fontsize=10)
        ax_sp.grid(alpha=0.3, which="both")
        hs, ls = [], []
        for a_ in (ax_sp, ax_sp2):
            h_, l_ = a_.get_legend_handles_labels()
            hs += h_
            ls += l_
        if hs:
            ax_sp.legend(hs, ls, fontsize=7 if many else 8, loc="upper right")
        cur_sp.draw(series, on=v_spcur.get(), xlabel="E (keV)")
        fig_sp.tight_layout()
        canvas_sp.draw_idle()

    cur_sp = XCursor(canvas_sp, ax_sp, toolbar_sp, "E (keV)")
    cur_bg = XCursor(canvas_bg, ax_bg, toolbar_bg, "depth")
    # depth unit of each stopping unit (energy part, depth part): for the
    # Bragg peak, x = integral of dE / S(E)
    SP_DEPTH = {1: (1e3, "Å"), 2: (1.0, "µm"), 3: (1e-3, "mm"),
                4: (1.0, "µg/cm²"), 5: (1e-3, "mg/cm²"),
                6: (1.0, "mg/cm²"), 7: (1e3, "10¹⁵ at/cm²")}

    def sp_bragg():
        """Bragg peak: total stopping S(E) of each table along the ion
        path, from E0 down to the lowest energy of the table:
        x(E) = integral from E to E0 of dE'/S(E') (S interpolated in
        log-log), plotted as S vs x; the depth unit is the one implied by
        the stopping units (e.g. eV/Å -> Å, eV/(1e15 at/cm2) ->
        1e15 at/cm2)."""
        ax_bg.clear()
        tables = sp_data["tables"]
        msg = ""
        try:
            e0 = float(v_bge0.get().replace(",", "."))
        except ValueError:
            e0 = None
        dep = SP_DEPTH.get(sp_data.get("unit_no"))
        series = []
        if not tables:
            msg = "Run SR (or load an SR output file) first."
        elif dep is None:
            msg = "Bragg peak: not with L.S.S. reduced units."
        elif not e0 or e0 <= 0:
            msg = "Type the initial energy E₀ (keV)."
        else:
            emult, dunit = dep          # keV -> energy unit of S
            info = []
            for i_, (title, t_) in enumerate(tables):
                e = np.asarray(t_["energy"], float)
                s = np.asarray(t_["elec"] + t_["nucl"], float)
                ok = (e > 0) & (s > 0)
                e, s = e[ok], s[ok]
                if len(e) < 2 or e0 <= e.min():
                    info.append(f"{title}: E₀ below the table")
                    continue
                if e0 > e.max():
                    info.append(f"{title}: E₀ above the table "
                                f"({e.max():g} keV), not drawn")
                    continue
                # fine energy grid from E0 down to the lowest energy
                eg = np.geomspace(e0, e.min(), 2000)
                sg = np.exp(np.interp(np.log(eg), np.log(e), np.log(s)))
                # x = integral of dE/S (E in the energy unit of S)
                de = -np.diff(eg) * emult
                # trapezoids of 1/S over each energy step
                x = np.concatenate([[0.0], np.cumsum(
                    de * 0.5 * (1 / sg[:-1] + 1 / sg[1:]))])
                col = PALETTE[i_ % len(PALETTE)]
                lab = title if len(tables) > 1 else "Total stopping"
                ax_bg.plot(x, sg, color=col, lw=1.8, label=lab)
                k_pk = int(np.argmax(sg))
                ax_bg.axvline(x[k_pk], color=col, ls=":", lw=1)
                series.append((lab, x, sg, col, ax_bg))
                info.append(f"{title}: peak {sg[k_pk]:.4g} at "
                            f"{x[k_pk]:.4g} {dunit}, path "
                            f"{x[-1]:.4g} {dunit} (to {e.min():g} keV)")
            ax_bg.set_xlabel(f"Depth along the path ({dunit})")
            ax_bg.set_ylabel(f"Total stopping ({sp_data['units']})")
            ax_bg.set_title(f"Bragg curve, E₀ = {e0:g} keV", fontsize=10)
            ax_bg.grid(alpha=0.3)
            if series:
                ax_bg.legend(fontsize=8, loc="lower left")
            msg = "; ".join(info)
            cur_bg.draw(series, on=v_spcur.get(), xlabel=f"x ({dunit})")
        if not series:
            ax_bg.text(0.5, 0.5, msg or "No curve", transform=ax_bg.transAxes,
                       ha="center", color="#888888", wrap=True)
        lbl_bg.config(text=msg if series else "")
        fig_bg.tight_layout()
        canvas_bg.draw_idle()

    def sp_export():
        tables = sp_data["tables"]
        if not tables:
            messagebox.showinfo("Export table", "No table yet.")
            return
        p_ = filedialog.asksaveasfilename(
            title="Export the stopping and range table(s)",
            defaultextension=".txt", initialfile="stopping_range.txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not p_:
            return
        u_ = sp_data["units"]
        ru, div = v_sprunit.get(), SP_RDIV[v_sprunit.get()]
        lines = ["# SRIM SR Module tables, exported by ionolum",
                 f"# Stopping units = {u_}", f"# Range units = {ru}"]
        for title, t_ in tables:
            lines += ["", f"# {title}",
                      f"# E(keV)\tdEdx_elec\tdEdx_nucl\tdEdx_total\t"
                      f"Range({ru})\tLong_straggling({ru})\t"
                      f"Lat_straggling({ru})"]
            for k in range(len(t_["energy"])):
                lines.append("\t".join(f"{x:.6g}" for x in (
                    t_["energy"][k], t_["elec"][k], t_["nucl"][k],
                    t_["elec"][k] + t_["nucl"][k], t_["range"][k] / div,
                    t_["long"][k] / div, t_["lat"][k] / div)))
        try:
            with open(p_, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            messagebox.showerror("Export table", str(e))
            return
        lbl_sp.config(text=f"Table(s) exported to {p_}",
                      foreground="#555555")

    def on_srim_subtab(_e=None):
        trimin_area.pack_forget()
        res_area.pack_forget()
        sp_area.pack_forget()
        cur = sub_s.index(sub_s.select())
        if cur == sub_s.index(tab_ss_outer):
            trimin_area.pack(fill="both", expand=True)
            srim_refresh()
        elif cur == sub_s.index(tab_sp):
            sp_area.pack(fill="both", expand=True)
            srim_refresh()                      # layers of Structure
            sp_refresh()
            sp_plot()
            sp_bragg()
        else:
            res_area.pack(fill="both", expand=True)
            sres_plot()
    sub_s.bind("<<NotebookTabChanged>>", on_srim_subtab)
    # the layers are read again each time the SRIM tab is opened
    nb.bind("<<NotebookTabChanged>>", lambda e: nb.select() == str(tab_s)
            and srim_refresh(), add="+")
    srim_refresh()

    def set_ion(sym, mass, energy_kev):
        """Ion of Structure from the Experimental setup of ionolum (energy at the sample)."""
        if sr["ion"].get().strip() != sym:
            sr["ion"].set(sym)                 # on_sion: list of the isotopes
        sr["mass"].set(f"{mass:.4f}")
        sr["energy"].set(f"{energy_kev:.1f}")
    hooks["setion"] = set_ion

    def srim_state():
        """State of SRIM › Structure (ion, options, target layers, atoms, TRIM.IN) for a session."""
        def keys(d_):
            return {str(k): v for k, v in d_.items()}
        return {"fields": {k: v.get() for k, v in sr.items()},
                "flags": {k: v.get() for k, v in sr_flags.items()},
                "auto_trim_in": v_trim_auto.get(),
                "trim_in": txt_trimin.get("1.0", "end-1c"),
                "extra": keys(srim_extra), "overrides": keys(srim_ovr), "rho": keys(srim_rho),
                "gas": keys(srim_gas), "removed": sorted(str(k) for k in srim_removed),
                "order": [str(k) for k in srim_order], "atoms": dict(srim_atoms),
                "results_folder": v_sres.get()}

    def srim_restore(st):
        """SRIM › Structure from srim_state (a session)."""
        def key(k):
            return int(k) if str(k).isdigit() else str(k)
        for k, v in (st.get("fields") or {}).items():
            if k in sr:
                sr[k].set(v)
        for k, v in (st.get("flags") or {}).items():
            if k in sr_flags:
                sr_flags[k].set(bool(v))
        for d_, name in ((srim_extra, "extra"), (srim_ovr, "overrides"), (srim_rho, "rho"),
                         (srim_gas, "gas")):
            d_.clear()
            d_.update((key(k), v) for k, v in (st.get(name) or {}).items())
        srim_removed.clear()
        srim_removed.update(key(k) for k in st.get("removed", []))
        srim_order[:] = [key(k) for k in st.get("order", [])]
        srim_atoms.clear()
        srim_atoms.update((k, list(v)) for k, v in (st.get("atoms") or {}).items())
        if st.get("results_folder"):
            v_sres.set(st["results_folder"])
        auto = bool(st.get("auto_trim_in", True))
        v_trim_auto.set(auto)
        srim_refresh()
        if not auto:                           # TRIM.IN edited by hand: the text saved
            txt_trimin.delete("1.0", "end")
            txt_trimin.insert("1.0", st.get("trim_in", ""))
            txt_trimin.edit_reset()
    hooks["srim_state"] = srim_state
    hooks["srim_restore"] = srim_restore
