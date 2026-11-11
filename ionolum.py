"""Ionoluminescence data treatment (GUI).

A data group = the files of one experiment in one folder:
  NAME_ionospectra<n>.txt          spectra acquired along the time. Line 1: "Total Time: <ms>",
                                   "Beam Charge", "Beam Current" (not valid, ignored), "X: Y: Z: x y z",
                                   "Date <date>"; then 2 columns: wavelength (nm), intensity
  NAME_current_K6485_esp_<n>.txt   beam current (A, Keithley 6485) read during the integration of
                                   each spectrum (one line of readings, ~10 ms each)
The spectra and the current files are paired by the order of their numbers (ionospectra01 <->
esp_0, ionospectra11 <-> esp_1, ...).

    python ionolum.py

The functions are in the package ionolum_lib (next to this file).
"""

import json
import os
import re
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# the package ionolum_lib is next to this file, wherever the program is installed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ionolum_lib.constants import (  # noqa: E402
    AIR_LINES, AIR_LINES_MIN, ATMOSPHERES, FLUENCE_UNIT, LOWER_PLOTS, NORM_MODES, SRIM_DIR, TAB10)
from ionolum_lib.data_files import (  # noqa: E402
    NOBG_RE, background_of, hms, info_number, load_group, nobg_name, read_area_file, read_nobg, split_name,
    write_nobg)
from ionolum_lib.beam import (  # noqa: E402
    beam_at_sample, beam_spot, fluence_per_coulomb, gas_density, run_trim_transmit, trim_run_dir)
from ionolum_lib.models import DEPTH_UNITS, MODEL_PRESETS, X_UNITS, eval_model, trapz  # noqa: E402


# ---------------------------------------------------------------- GUI
class App:
    def __init__(self, root):
        self.root = root
        root.title("Ionoluminescence")
        root.geometry("1350x820")
        self.items = []
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)
        tab = ttk.Frame(nb)
        nb.add(tab, text="Import data")
        self.build_import(tab)
        tab = ttk.Frame(nb)
        nb.add(tab, text="Data treatment")
        self.build_treat(tab)
        self.nb = nb
        nb.bind("<<NotebookTabChanged>>", self.on_tab, add="+")
        tab = ttk.Frame(nb)
        nb.add(tab, text="Experimental setup")
        self.build_setup(tab)
        self.hooks = {}
        try:
            from ionolum_lib.srim_tab import build_srim_tab
            build_srim_tab(root, nb, self.hooks)
        except Exception as e:          # e.g. pysrim not found: the rest still works
            tab = ttk.Frame(nb)
            nb.add(tab, text="  SRIM  ")
            ttk.Label(tab, text="The SRIM tab could not be built:\n%s" % e, foreground="#c00").pack(padx=20, pady=20)

    def build_import(self, tab):
        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, width=430)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        # ---- data group
        g = ttk.LabelFrame(left, text="Data group")
        g.pack(fill="x", padx=6, pady=6)
        fs = ttk.Frame(g)
        fs.grid(row=0, column=0, columnspan=3, sticky="we", padx=6, pady=(6, 4))
        ttk.Button(fs, text="Select file…", command=self.select_file).pack(side="left")
        ttk.Button(fs, text="Import session…", command=self.import_session).pack(side="right")
        ttk.Button(fs, text="Save session…", command=self.save_session).pack(side="right", padx=4)
        ttk.Label(g, text="Folder:").grid(row=1, column=0, sticky="w", padx=6)
        self.v_dir = tk.StringVar()
        ttk.Entry(g, textvariable=self.v_dir, width=44).grid(row=1, column=1, columnspan=2, sticky="we", padx=(0, 6))
        ttk.Label(g, text="Name:").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        self.v_name = tk.StringVar()
        ttk.Entry(g, textvariable=self.v_name, width=30).grid(row=2, column=1, sticky="we", pady=2)
        ttk.Button(g, text="Load", width=7, command=self.load).grid(row=2, column=2, padx=6)
        ttk.Label(g, text="files: NAME_ionospectra<n>.txt and NAME_current_K6485_esp_<n>.txt",
                  foreground="#666").grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))
        g.columnconfigure(1, weight=1)

        inf = ttk.LabelFrame(left, text="Experiment")
        inf.pack(fill="x", padx=6, pady=(0, 6))
        self.inf = {}
        for r, (k, lab) in enumerate((("nspec", "Spectra imported"), ("ncurr", "Current files"),
                                      ("tspec", "Time per spectrum"), ("ttot", "Acquisition time"),
                                      ("xyz", "Position X Y Z"), ("date", "Date"))):
            ttk.Label(inf, text=lab + ":").grid(row=r, column=0, sticky="w", padx=6, pady=1)
            self.inf[k] = ttk.Label(inf, text="—")
            self.inf[k].grid(row=r, column=1, sticky="w", padx=6, pady=1)

        # ---- background (one spectrum of the group, by default the last one)
        bgf = ttk.LabelFrame(left, text="Background")
        bgf.pack(fill="x", padx=6, pady=(0, 6))
        self.v_bgon = tk.BooleanVar(value=False)
        ttk.Checkbutton(bgf, text="Remove background", variable=self.v_bgon,
                        command=self.draw).grid(row=0, column=0, sticky="w", padx=6, pady=(4, 2))
        ttk.Label(bgf, text="spectrum #").grid(row=0, column=1, sticky="e", padx=(6, 2))
        self.v_bg = tk.StringVar()
        self.sp_bg = tk.Spinbox(bgf, textvariable=self.v_bg, from_=1, to=1, width=6, command=self.bg_changed)
        self.sp_bg.grid(row=0, column=2, sticky="w")
        self.sp_bg.bind("<KeyRelease>", lambda e: self.bg_changed(delay=400))
        self.l_bg = ttk.Label(bgf, text="", foreground="#666")
        self.l_bg.grid(row=1, column=0, columnspan=3, sticky="w", padx=6)
        b_create = ttk.Button(bgf, command=self.create_nobg)
        b_create.grid(row=2, column=0, columnspan=3, sticky="w", padx=6, pady=(2, 2))

        def create_text(*_):
            b_create.configure(text="Create files without background (selected spectra)" if self.v_bgon.get()
                               else "Create files, background 0 (selected spectra)")
        self.v_bgon.trace_add("write", create_text)
        create_text()
        self.l_bgmsg = ttk.Label(bgf, text="", foreground="#070", wraplength=400, justify="left")
        self.l_bgmsg.grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))
        self.bg_job = None

        # ---- list of the spectra (only the ionospectra files)
        lf = ttk.LabelFrame(left, text="Spectra")
        lf.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        fr = ttk.Frame(lf)
        fr.pack(fill="both", expand=True, padx=4, pady=4)
        cols = (("k", "#", 40), ("file", "File", 190), ("t", "Start (s)", 65), ("i", "I mean (nA)", 80),
                ("q", "Q (nC)", 70), ("f", "Φ (10¹²/cm²)", 85))
        self.tv = ttk.Treeview(fr, columns=[c[0] for c in cols], show="headings", selectmode="extended")
        for c, t, w in cols:
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor="w" if c == "file" else "e", stretch=(c == "file"))
        sb = ttk.Scrollbar(fr, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tv.pack(side="left", fill="both", expand=True)
        self.tv.bind("<<TreeviewSelect>>", lambda e: self.draw())
        bf = ttk.Frame(lf)
        bf.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(bf, text="Select all", command=lambda: self.tv.selection_set(self.tv.get_children())
                   ).pack(side="left")
        ttk.Button(bf, text="Deselect all", command=lambda: self.tv.selection_set(())
                   ).pack(side="left", padx=4)
        ttk.Label(bf, text="Ctrl / Shift: several", foreground="#666").pack(side="left", padx=6)

        # ---- plots
        opt = ttk.Frame(right)
        opt.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(opt, text="Normalize to:").pack(side="left")
        self.v_normk = tk.StringVar(value=NORM_MODES[0])
        cb = ttk.Combobox(opt, textvariable=self.v_normk, values=NORM_MODES, state="readonly", width=33)
        cb.pack(side="left", padx=(4, 10))
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw())
        self.v_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Log intensity", variable=self.v_log, command=self.draw).pack(side="left")
        self.v_art = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Show artefacts (air N₂ lines)", variable=self.v_art,
                        command=self.draw).pack(side="left", padx=(10, 0))
        opt2 = ttk.Frame(right)
        opt2.pack(fill="x", padx=6, pady=(2, 0))
        self.v_cur = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt2, text="Lower plot:", variable=self.v_cur, command=self.draw).pack(side="left")
        self.v_low = tk.StringVar(value=LOWER_PLOTS[0])
        cb = ttk.Combobox(opt2, textvariable=self.v_low, values=LOWER_PLOTS, state="readonly", width=24)
        cb.pack(side="left", padx=(4, 10))
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw())
        self.v_all = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt2, text="Whole trace in grey", variable=self.v_all,
                        command=self.draw).pack(side="left")
        self.l_flu = ttk.Label(opt2, text="", foreground="#666")
        self.l_flu.pack(side="left", padx=10)
        self.fig = Figure(figsize=(8, 6), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        tb = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.draw()

    # ------------------------------------------------ data treatment (the _nobg.txt files)
    def build_treat(self, tab):
        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="both", expand=True)
        outer = ttk.Frame(pw, width=430)
        right = ttk.Frame(pw)
        pw.add(outer, weight=0)
        pw.add(right, weight=1)
        self.tfiles = []                      # read_nobg of each file + "file", "n", "fluence", "elapsed"
        # sub-tabs: Data (the _nobg files, region of interest, area vs fluence) and Models (fit of the
        # saved area vs fluence curves, SRIM ionization and vacancies); the list of the saved area
        # vs fluence files is shown in both (packed in the container of the sub-tab shown)
        self.tsub = ttk.Notebook(outer)
        self.tsub.pack(fill="both", expand=True)
        left = ttk.Frame(self.tsub)
        tmod = ttk.Frame(self.tsub)
        self.tsub.add(left, text="  Data  ")
        self.tsub.add(tmod, text="  Models  ")

        g = ttk.LabelFrame(left, text="Files without background (NAME_ionospectra<n>_nobg.txt)")
        g.pack(fill="x", padx=6, pady=6)
        ttk.Button(g, text="Select file…", command=self.t_select).grid(row=0, column=0, columnspan=3,
                                                                      sticky="w", padx=6, pady=(6, 4))
        ttk.Label(g, text="Folder:").grid(row=1, column=0, sticky="w", padx=6)
        self.t_dir = tk.StringVar()
        ttk.Entry(g, textvariable=self.t_dir, width=44).grid(row=1, column=1, columnspan=2, sticky="we", padx=(0, 6))
        ttk.Label(g, text="Name:").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        self.t_name = tk.StringVar()
        ttk.Entry(g, textvariable=self.t_name, width=30).grid(row=2, column=1, sticky="we", pady=2)
        ttk.Button(g, text="Load", width=7, command=self.t_load).grid(row=2, column=2, padx=6)
        self.l_tmsg = ttk.Label(g, text="", foreground="#666", wraplength=400, justify="left")
        self.l_tmsg.grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))
        g.columnconfigure(1, weight=1)

        # region of interest: integration limits (moved also with the vertical lines of the lower plot)
        r = ttk.LabelFrame(left, text="Region of interest (integrated area)")
        r.pack(fill="x", padx=6, pady=(0, 6))
        self.t_x1, self.t_x2 = tk.StringVar(), tk.StringVar()
        self.t_ents = []
        self.t_busy = False
        for i, (lab, v) in enumerate((("From", self.t_x1), ("To", self.t_x2))):
            ttk.Label(r, text=lab).grid(row=0, column=2 * i, sticky="w", padx=(6, 2), pady=4)
            e = tk.Entry(r, textvariable=v, width=10, justify="right")
            e.grid(row=0, column=2 * i + 1, sticky="w", pady=4)
            self.t_ents.append(e)
            v.trace_add("write", lambda *a: self.t_roi_typed())
        ttk.Label(r, text="nm  (or drag the vertical lines)", foreground="#666").grid(row=0, column=4, sticky="w",
                                                                                     padx=6)
        ttk.Button(r, text="Whole spectrum", command=self.t_roi_all).grid(row=1, column=0, columnspan=3, sticky="w",
                                                                         padx=6, pady=(0, 4))
        ttk.Button(r, text="Save area vs fluence…", command=self.t_save).grid(row=1, column=3, columnspan=2,
                                                                             sticky="w", padx=6, pady=(0, 4))

        self.t_svbox_data = ttk.Frame(left)           # place of the saved files list in Data
        self.t_svbox_data.pack(side="bottom", fill="x", padx=6, pady=(0, 6))
        lf = ttk.LabelFrame(left, text="Files")
        lf.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        pv = ttk.PanedWindow(lf, orient="vertical")
        pv.pack(fill="both", expand=True, padx=4, pady=4)
        fr = ttk.Frame(pv)
        cols = (("k", "#", 40), ("file", "File", 200), ("t", "Elapsed (s)", 75), ("f", "Φ acc. (ions/cm²)", 110),
                ("a", "Area", 80))
        self.ttv = ttk.Treeview(fr, columns=[c[0] for c in cols], show="headings", selectmode="extended")
        for c, t, w in cols:
            self.ttv.heading(c, text=t)
            self.ttv.column(c, width=w, anchor="w" if c == "file" else "e", stretch=(c == "file"))
        sb = ttk.Scrollbar(fr, orient="vertical", command=self.ttv.yview)
        self.ttv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.ttv.pack(side="left", fill="both", expand=True)
        self.ttv.bind("<<TreeviewSelect>>", lambda e: self.t_selected())
        # a click on a file already selected (no <<TreeviewSelect>>) also leaves the saved files view
        self.ttv.bind("<ButtonRelease-1>", lambda e: self.t_mode == "saved" and self.ttv.selection()
                      and self.t_selected())
        # disabled spectra (red): not used in the integrated area vs fluence nor saved with it
        self.ttv.tag_configure("disabled", foreground="#d00000", background="#fde4e4")
        self.t_disabled = set()                # file names
        inf = ttk.Frame(pv)
        bf = ttk.Frame(inf)
        bf.pack(fill="x", pady=(2, 0))
        ttk.Button(bf, text="Select all", command=lambda: self.ttv.selection_set(self.ttv.get_children())
                   ).pack(side="left")
        ttk.Button(bf, text="Deselect all", command=lambda: self.ttv.selection_set(())).pack(side="left", padx=4)
        bf = ttk.Frame(inf)
        bf.pack(fill="x", pady=(2, 2))
        ttk.Button(bf, text="Disable spectra", command=lambda: self.t_enable(False)).pack(side="left")
        ttk.Button(bf, text="Enable spectra", command=lambda: self.t_enable(True)).pack(side="left", padx=4)
        ttk.Button(bf, text="Enable all", command=lambda: self.t_enable(True, every=True)).pack(side="left")
        # in its place, the operations on the saved files while they are shown (t_show_panel)
        fi = self.t_finfo = ttk.Frame(inf)
        fi.pack(fill="both", expand=True)
        ttk.Label(fi, text="Line 1 of the selected file:").pack(anchor="w")
        self.t_info = tk.Text(fi, height=10, width=50, wrap="word", font=("Consolas", 9), relief="flat",
                              background="#f4f4f4")
        self.t_info.pack(fill="both", expand=True)
        self.t_info.configure(state="disabled")
        fo = self.t_fops = ttk.LabelFrame(inf, text="Saved area vs fluence curves")
        self.t_snorm = tk.BooleanVar(value=False)
        ttk.Checkbutton(fo, text="Normalize each curve to its maximum", variable=self.t_snorm,
                        command=self.t_draw).grid(row=0, column=0, columnspan=4, sticky="w", padx=6, pady=(4, 6))
        ttk.Label(fo, text="Value:").grid(row=1, column=0, sticky="w", padx=6)
        self.t_sval = tk.StringVar(value="1")
        ttk.Entry(fo, textvariable=self.t_sval, width=12, justify="right").grid(row=1, column=1, columnspan=3,
                                                                              sticky="w")
        for i, (txt, op) in enumerate((("Add", "+"), ("Subtract", "−"), ("Multiply", "×"), ("Divide", "÷"))):
            ttk.Button(fo, text=txt, width=9, command=lambda op=op: self.t_saved_op(op)).grid(
                row=2, column=i, sticky="w", padx=(6 if i == 0 else 2, 0), pady=4)
        ttk.Button(fo, text="Reset (original values)", command=lambda: self.t_saved_op("reset")).grid(
            row=3, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 4))
        self.l_sops = ttk.Label(fo, text="applied to the curves selected in the list below", foreground="#666",
                                wraplength=380, justify="left")
        self.l_sops.grid(row=4, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 4))
        # files written by Save area vs fluence: clicking one shows only it (upper plot)
        sv = self.t_svframe = ttk.Frame(outer)
        ttk.Label(sv, text="Saved area vs fluence files (click: show the file)").pack(anchor="w", pady=(4, 0))
        fs = ttk.Frame(sv)
        fs.pack(fill="both", expand=True)
        self.stv = ttk.Treeview(fs, columns=("pin", "colour", "file", "roi", "op"), show="headings", selectmode="extended",
                                height=4)
        # Show ☑: the curve is always drawn, also when other files are selected
        self.stv.heading("pin", text="Show")
        self.stv.column("pin", width=42, anchor="center", stretch=False)
        self.stv.bind("<Button-1>", self.t_pin_click)
        # Colour: a square of the colour of the curve in the plot (t_paint_colours)
        self.stv.heading("colour", text="Colour")
        self.stv.column("colour", width=50, anchor="center", stretch=False)
        self.stv.bind("<Configure>", lambda e: self.root.after_idle(self.t_paint_colours))
        self.stv.heading("file", text="File")
        self.stv.heading("roi", text="Region / spectra")
        self.stv.heading("op", text="Operation")
        self.stv.column("file", width=190, anchor="w")
        self.stv.column("roi", width=140, anchor="w")
        self.stv.column("op", width=90, anchor="w")
        sb = ttk.Scrollbar(fs, orient="vertical", command=self.stv.yview)
        self.stv.configure(yscrollcommand=lambda *a: (sb.set(*a), self.root.after_idle(self.t_paint_colours)))
        sb.pack(side="right", fill="y")
        self.stv.pack(side="left", fill="both", expand=True)
        self.stv.bind("<<TreeviewSelect>>", lambda e: self.t_saved_selected())
        bf = ttk.Frame(sv)
        bf.pack(fill="x", pady=(2, 0))
        ttk.Button(bf, text="Add file(s)…", command=self.t_saved_add_dialog).pack(side="left")
        ttk.Button(bf, text="Remove from list", command=self.t_saved_remove).pack(side="left", padx=4)
        self.t_saved = {}                     # iid -> read_area_file
        self.t_mode = "files"                 # "files": both plots; "saved": the saved files (upper only)
        pv.add(fr, weight=3)
        pv.add(inf, weight=2)
        self.build_models(tmod)
        sv.lift()                             # above the sub-tabs, where it is packed
        self.t_sub = "data"
        self.t_place_saved()
        self.tsub.bind("<<NotebookTabChanged>>", self.t_subtab)

        # ---- plots: integrated area vs fluence (top), spectra (bottom)
        opt = ttk.Frame(right)
        opt.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(opt, text="Spectra:").pack(side="left")
        self.t_col = tk.StringVar()
        self.cb_tcol = ttk.Combobox(opt, textvariable=self.t_col, state="readonly", width=55)
        self.cb_tcol.pack(side="left", padx=(4, 10))
        self.cb_tcol.bind("<<ComboboxSelected>>", lambda e: self.t_draw())
        self.t_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Log intensity", variable=self.t_log, command=self.t_draw).pack(side="left")
        self.tfig = Figure(figsize=(8, 6), dpi=100, layout="constrained")
        self.tcanvas = FigureCanvasTkAgg(self.tfig, master=right)
        self.ttb = NavigationToolbar2Tk(self.tcanvas, right, pack_toolbar=False)
        self.ttb.update()
        self.ttb.pack(side="bottom", fill="x")
        self.tcanvas.get_tk_widget().pack(fill="both", expand=True)
        self.t_lines, self.t_drag, self.t_busy, self.t_job = [], None, False, None
        self.t_cur_x, self.t_cline, self.t_ctext = None, None, None     # green cursor (lower plot)
        self.t_scolors = {}                   # saved curve iid -> colour drawn (column Colour)
        self.t_swatches = []
        self.tcanvas.mpl_connect("button_press_event", self.t_press)
        self.tcanvas.mpl_connect("motion_notify_event", self.t_motion)
        self.tcanvas.mpl_connect("button_release_event", lambda e: setattr(self, "t_drag", None))
        self.t_draw()

    def t_place_saved(self):
        """The list of the saved area vs fluence files in the sub-tab shown."""
        box = self.m_svbox if self.t_sub == "models" else self.t_svbox_data
        self.t_svframe.pack_forget()
        self.t_svframe.pack(in_=box, fill="both", expand=True)
        self.stv.configure(height=8 if self.t_sub == "models" else 4)

    def t_subtab(self, _e=None):
        self.t_sub = "models" if self.tsub.index(self.tsub.select()) == 1 else "data"
        self.t_place_saved()
        self.t_draw()

    MODEL_PRESETS, X_UNITS, DEPTH_UNITS = MODEL_PRESETS, X_UNITS, DEPTH_UNITS     # (models.py)

    def build_models(self, tab):
        self.m_svbox = ttk.LabelFrame(tab, text="Area vs fluence curves (select the ones to fit)")
        self.m_svbox.pack(fill="both", expand=True, padx=6, pady=6)

        f = ttk.LabelFrame(tab, text="Fit model  y(x),  x = fluence")
        f.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(f, text="Model").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self.m_preset = tk.StringVar(value="Birks–Black  a0 / (1 + a1·x)")
        cb = ttk.Combobox(f, textvariable=self.m_preset, values=list(self.MODEL_PRESETS), state="readonly", width=34)
        cb.grid(row=0, column=1, columnspan=5, sticky="w", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self.m_preset_chosen())
        ttk.Label(f, text="y(x) =").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        self.m_expr = tk.StringVar(value=self.MODEL_PRESETS[self.m_preset.get()])
        ttk.Entry(f, textvariable=self.m_expr, width=44).grid(row=1, column=1, columnspan=5, sticky="we", pady=2,
                                                            padx=(0, 6))
        self.m_expr.trace_add("write", lambda *a: self.m_expr_typed())
        ttk.Label(f, text="x in").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        self.m_xu = tk.StringVar(value="10¹⁵ ions/cm²")
        cb = ttk.Combobox(f, textvariable=self.m_xu, values=list(self.X_UNITS), state="readonly", width=14)
        cb.grid(row=2, column=1, columnspan=2, sticky="w", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self.m_clear_fit())
        ttk.Label(f, text="parameters a0 … a9; exp, log, log10, sqrt, sin, cos, tanh, abs; ^ or ** powers",
                  foreground="#666", wraplength=400).grid(row=3, column=0, columnspan=6, sticky="w", padx=6)
        pf = ttk.Frame(f)
        pf.grid(row=4, column=0, columnspan=6, sticky="w", padx=6, pady=4)
        for j, h in enumerate(("", "Value", "Min", "Max", "Fit", "±")):
            ttk.Label(pf, text=h, foreground="#555").grid(row=0, column=j, padx=2)
        self.m_par = []
        for k in range(10):
            row = {"w": [], "v": tk.StringVar(value="1"), "min": tk.StringVar(), "max": tk.StringVar(),
                   "fit": tk.BooleanVar(value=True)}
            row["w"].append(ttk.Label(pf, text="a%d" % k))
            for key, w in (("v", 12), ("min", 8), ("max", 8)):
                row["w"].append(ttk.Entry(pf, textvariable=row[key], width=w, justify="right"))
            row["w"].append(ttk.Checkbutton(pf, variable=row["fit"]))
            row["err"] = ttk.Label(pf, text="", width=10)
            row["w"].append(row["err"])
            for j, w in enumerate(row["w"]):
                w.grid(row=k + 1, column=j, padx=2, pady=1, sticky="w")
            self.m_par.append(row)
        bf = ttk.Frame(f)
        bf.grid(row=5, column=0, columnspan=6, sticky="w", padx=6, pady=(0, 4))
        ttk.Button(bf, text="Initial guess", command=self.m_guess).pack(side="left")
        ttk.Button(bf, text="Fit selected curves", command=self.m_fit).pack(side="left", padx=4)
        ttk.Button(bf, text="Clear fit", command=self.m_clear_fit).pack(side="left")
        self.m_res = tk.Text(f, height=5, width=50, wrap="word", font=("Consolas", 9), relief="flat",
                             background="#f4f4f4")
        self.m_res.grid(row=6, column=0, columnspan=6, sticky="we", padx=6, pady=(0, 6))
        self.m_fits = {}                       # saved curve iid -> (expression, x unit, parameters)

        s = ttk.LabelFrame(tab, text="SRIM simulations (lower plots: ionization, vacancies)")
        s.pack(fill="x", padx=6, pady=(0, 6))
        fs = ttk.Frame(s)
        fs.pack(fill="x", padx=4, pady=4)
        self.mtv = ttk.Treeview(fs, columns=("file", "kind", "dir"), show="headings", height=4, selectmode="extended")
        for c, t, w in (("file", "File", 170), ("kind", "Type", 70), ("dir", "Folder", 160)):
            self.mtv.heading(c, text=t)
            self.mtv.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(fs, orient="vertical", command=self.mtv.yview)
        self.mtv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.mtv.pack(side="left", fill="x", expand=True)
        bf = ttk.Frame(s)
        bf.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(bf, text="Load files…", command=self.m_srim_add_dialog).pack(side="left")
        ttk.Button(bf, text="Load folder…", command=self.m_srim_add_folder).pack(side="left", padx=4)
        ttk.Button(bf, text="Remove", command=self.m_srim_remove).pack(side="left")
        ttk.Label(bf, text="Depth in").pack(side="left", padx=(10, 2))
        self.m_du = tk.StringVar(value="µm")
        cb = ttk.Combobox(bf, textvariable=self.m_du, values=list(self.DEPTH_UNITS), state="readonly", width=4)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.t_draw())
        self.l_msrim = ttk.Label(s, text="IONIZ*.txt and VACANCY*.txt of TRIM runs (also with a prefix)",
                                 foreground="#666", wraplength=400)
        self.l_msrim.pack(anchor="w", padx=6, pady=(0, 4))
        self.m_srim = {}                       # iid -> {"path", "kind", "data"}
        self.m_expr_typed()

    def m_used(self):
        """Indices of the parameters a0 … a9 in the expression."""
        return sorted({int(k) for k in re.findall(r"(?<![A-Za-z_0-9])a(\d)(?![0-9A-Za-z_])", self.m_expr.get())})

    def m_expr_typed(self):
        used = self.m_used()
        for k, row in enumerate(self.m_par):
            for w in row["w"]:
                (w.grid if k in used else w.grid_remove)()
        if self.m_expr.get().strip() != self.MODEL_PRESETS.get(self.m_preset.get(), None):
            if self.m_preset.get() != "Custom" and hasattr(self, "m_res"):
                self.m_preset.set("Custom")

    def m_preset_chosen(self):
        e = self.MODEL_PRESETS[self.m_preset.get()]
        if e:
            self.m_expr.set(e)
            self.m_guess(quiet=True)

    def m_curves(self):
        """(iid, x in the unit of the model, y as drawn) of the saved curves selected."""
        out = []
        xu = self.X_UNITS[self.m_xu.get()]
        for iid in self.stv.selection():
            d = self.t_saved[iid]
            x, y = d["fluence"] / xu, self.t_curve_y(d)
            m = np.isfinite(x) & np.isfinite(y)
            out.append((iid, x[m], y[m]))
        return out

    def m_guess(self, quiet=False):
        """Initial values of the parameters of the preset models from the first curve selected."""
        cs = self.m_curves()
        if not cs or not len(cs[0][1]):
            if not quiet:
                self.m_write("Select a curve in the list to guess the parameters.")
            return
        x, y = cs[0][1], cs[0][2]
        e = self.m_expr.get().replace(" ", "")
        y0, ye = float(y[np.argmin(x)]), float(y[np.argmax(x)])
        span = float(np.ptp(x)) or 1.0
        # x where y falls to the middle of its range: for the decay constants
        half = float(x[np.argmin(np.abs(y - (y0 + ye) / 2))]) or span / 2
        g = {"a0/(1+a1*x)": [y0, 1.0 / half],
             # k = 1: half of L0 − C at x = ln 2 / σ (the middle of the decay)
             "a0/(1+a1*(exp(a2*x)-1))+a3": [y0 - ye, 1.0, np.log(2) / half, ye],
             "a0*exp(-a1*x)+a2": [y0 - ye, np.log(2) / half, ye],
             "a0*exp(-a1*x)+a2*exp(-a3*x)+a4": [(y0 - ye) / 2, 3 * np.log(2) / half, (y0 - ye) / 2,
                                                np.log(2) / (3 * half), ye],
             "a0+a1*x": list(np.polyfit(x, y, 1)[::-1]) if len(x) > 1 else [y0, 0.0]}.get(e)
        if e == "a0*x**a1":
            m = (x > 0) & (y > 0)
            if m.sum() > 1:
                b, a = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
                g = [float(np.exp(a)), float(b)]
        if g is None:
            if not quiet:
                self.m_write("No automatic guess for a custom model: type the initial values.")
            return
        for k, v in enumerate(g):
            self.m_par[k]["v"].set("%.6g" % v)
            self.m_par[k]["err"].configure(text="")

    def m_write(self, txt):
        self.m_res.delete("1.0", "end")
        self.m_res.insert("end", txt)

    def m_fit(self):
        """Least squares fit (scipy, trf; min/max = bounds) of the model to each selected curve."""
        from scipy.optimize import curve_fit
        expr, used = self.m_expr.get().strip(), self.m_used()
        cs = self.m_curves()
        if not cs:
            self.m_write("Select the curve(s) to fit in the list above.")
            return
        try:
            p0 = [float(self.m_par[k]["v"].get()) for k in range(10)]
            lo = [float(self.m_par[k]["min"].get()) if self.m_par[k]["min"].get().strip() else -np.inf
                  for k in range(10)]
            hi = [float(self.m_par[k]["max"].get()) if self.m_par[k]["max"].get().strip() else np.inf
                  for k in range(10)]
            eval_model(expr, p0, np.array([1.0]))
        except ValueError as e:
            self.m_write("Model or parameters: %s" % e)
            return
        free = [k for k in used if self.m_par[k]["fit"].get()]
        if not free:
            self.m_write("No parameter to fit (tick Fit).")
            return
        # free parameters relative to their initial values (scales like 1e5 and 1e-3 together)
        sc = np.array([abs(p0[k]) or 1.0 for k in free])
        out, last = [], None
        for iid, x, y in cs:
            name = os.path.basename(self.t_saved[iid]["path"])
            if len(x) <= len(free):
                out.append("%s: too few points" % name)
                continue

            def f(xx, *u):
                p = list(p0)
                for k, v, s in zip(free, u, sc):
                    p[k] = v * s
                return eval_model(expr, p, xx)
            u0 = np.clip(np.array([p0[k] for k in free]) / sc, np.array([lo[k] for k in free]) / sc,
                         np.array([hi[k] for k in free]) / sc)
            try:
                u, cov = curve_fit(f, x, y, p0=u0, bounds=(np.array([lo[k] for k in free]) / sc,
                                                           np.array([hi[k] for k in free]) / sc),
                                   method="trf", maxfev=20000)
            except Exception as e:
                out.append("%s: fit failed (%s)" % (name, e))
                continue
            p = list(p0)
            err = [None] * 10
            sd = np.sqrt(np.clip(np.diag(cov), 0, None)) if np.all(np.isfinite(cov)) else [np.nan] * len(free)
            for k, v, s, e in zip(free, u, sc, sd):
                p[k], err[k] = v * s, e * s
            r = y - eval_model(expr, p, x)
            r2 = 1 - np.sum(r ** 2) / np.sum((y - np.mean(y)) ** 2) if np.ptp(y) else np.nan
            self.m_fits[iid] = (expr, self.m_xu.get(), p)
            out.append("%s  (R² = %.5f)\n  " % (name, r2) + "\n  ".join(
                "a%d = %.6g%s" % (k, p[k], " ± %.3g" % err[k] if err[k] is not None else " (fixed)") for k in used))
            xu = self.X_UNITS[self.m_xu.get()]
            e0 = expr.replace(" ", "")
            if e0 == "a0/(1+a1*x)" and p[1]:
                out.append("  F½ = 1/a1 = %.4g ions/cm²" % (xu / p[1]))
            elif e0 == "a0/(1+a1*(exp(a2*x)-1))+a3":
                s_ = p[2] / xu
                # L − C = L0/2 at k (exp(σF) − 1) = 1
                out.append("  k = %.4g;  σ = %.4g cm² (%.4g Å²)%s" % (
                    p[1], s_, s_ * 1e16, ";  F½ = ln(1 + 1/k)/σ = %.4g ions/cm²" % (np.log(1 + 1 / p[1]) / s_)
                    if p[1] > 0 and s_ > 0 else ""))
            last = (p, err)
        if len(cs) == 1 and last:
            for k in used:
                self.m_par[k]["v"].set("%.6g" % last[0][k])
                self.m_par[k]["err"].configure(text="± %.3g" % last[1][k] if last[1][k] is not None else "")
        self.m_write("x = fluence in %s\n" % self.m_xu.get() + "\n".join(out))
        self.t_draw()

    def m_clear_fit(self):
        self.m_fits = {}
        for row in self.m_par:
            row["err"].configure(text="")
        self.m_write("")
        self.t_draw()

    # --- SRIM files
    def m_srim_add(self, path):
        from ionolum_lib.srim_tools import srim_known, srim_table
        p = os.path.normpath(path)
        kind = srim_known(os.path.basename(p))[0]
        if kind not in ("IONIZ.txt", "VACANCY.txt"):
            return "not an IONIZ / VACANCY file: %s" % os.path.basename(p)
        if any(os.path.normcase(d["path"]) == os.path.normcase(p) for d in self.m_srim.values()):
            return None
        try:
            data = srim_table(p)
        except Exception as e:
            return "%s: %s" % (os.path.basename(p), e)
        iid = "m%d" % (max([int(i[1:]) for i in self.m_srim] or [0]) + 1)
        self.m_srim[iid] = {"path": p, "kind": kind, "data": data}
        self.mtv.insert("", "end", iid=iid, values=(os.path.basename(p), kind.split(".")[0], os.path.dirname(p)))
        return None

    def m_srim_add_paths(self, paths):
        errs = [e for e in (self.m_srim_add(p) for p in paths) if e]
        self.l_msrim.configure(text="\n".join(errs) if errs else "%d SRIM file(s)" % len(self.m_srim),
                               foreground="#c00" if errs else "#666")
        self.t_draw()

    def m_srim_add_dialog(self):
        ps = filedialog.askopenfilenames(title="SRIM IONIZ / VACANCY files",
                                         initialdir=self.sv["srim"].get() if hasattr(self, "sv") else None,
                                         filetypes=[("SRIM ionization / vacancies", "*IONIZ*.txt *VACANCY*.txt"),
                                                    ("Text", "*.txt"), ("All files", "*.*")])
        if ps:
            self.m_srim_add_paths(ps)

    def m_srim_add_folder(self):
        from ionolum_lib.srim_tools import srim_known
        d = filedialog.askdirectory(title="Folder of a TRIM run (IONIZ / VACANCY files)")
        if not d:
            return
        ps = [os.path.join(d, f) for f in sorted(os.listdir(d))
              if srim_known(f)[0] in ("IONIZ.txt", "VACANCY.txt")]
        if not ps:
            self.l_msrim.configure(text="No IONIZ / VACANCY file in %s" % d, foreground="#c00")
            return
        self.m_srim_add_paths(ps)

    def m_srim_remove(self):
        for iid in self.mtv.selection():
            self.mtv.delete(iid)
            self.m_srim.pop(iid, None)
        self.t_draw()

    def m_draw(self):
        """Models: area vs fluence of the selected saved curves with the fits (top, 1/2 of the
        height), SRIM ionization and vacancies vs depth (1/4 each)."""
        f = self.tfig
        gs = f.add_gridspec(4, 1)
        ax = f.add_subplot(gs[0:2])
        axi = f.add_subplot(gs[2])
        axv = f.add_subplot(gs[3], sharex=axi)
        self.t_axl, self.t_lines = None, []
        sel = self.t_shown_saved()
        for i, iid in enumerate(sel):
            d = self.t_saved[iid]
            c = self.t_scolors[iid] = TAB10[i % len(TAB10)]
            ax.plot(d["fluence"], self.t_curve_y(d), "o", color=c, ms=4, label=os.path.basename(d["path"]))
            fit = self.m_fits.get(iid)
            if fit:
                xu = self.X_UNITS[fit[1]]
                xs = np.linspace(np.nanmin(d["fluence"]), np.nanmax(d["fluence"]), 300)
                try:
                    ax.plot(xs, eval_model(fit[0], fit[2], xs / xu), "--", color=c, lw=1.5, label="fit")
                except ValueError:
                    pass
        ax.set_xlabel("Fluence (ions/cm²)")
        ax.set_ylabel("Area / maximum" if self.t_snorm.get() else "Area", fontsize=8)
        ax.set_title("Area vs fluence: %s" % (self.m_expr.get().strip() or "no model"), fontsize=9)
        ax.grid(alpha=0.3)
        if sel:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "Select area vs fluence curves (left)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
        du = self.m_du.get()
        div = self.DEPTH_UNITS[du]
        ni = nv = 0
        for j, d in enumerate(self.m_srim.values()):
            a = d["data"]
            if a.ndim != 2 or a.shape[1] < 2:
                continue
            c = TAB10[j % len(TAB10)]
            name = os.path.basename(d["path"])
            x = a[:, 0] / div
            if d["kind"] == "IONIZ.txt":
                axi.plot(x, a[:, 1], color=c, lw=1.2, label=name + " ions")
                if a.shape[1] > 2:
                    axi.plot(x, a[:, 2], color=c, lw=1, ls="--", label=name + " recoils")
                ni += 1
            else:
                axv.plot(x, a[:, 1:].sum(axis=1), color=c, lw=1.2, label=name + " (total)")
                nv += 1
        axi.set_ylabel("Ionization\n(eV/(Å·ion))", fontsize=8)
        axv.set_ylabel("Vacancies\n(/(Å·ion))", fontsize=8)
        axv.set_xlabel("Depth (%s)" % du)
        for a_, n, kind in ((axi, ni, "IONIZ"), (axv, nv, "VACANCY")):
            a_.grid(alpha=0.3)
            if n:
                a_.legend(fontsize=7)
            else:
                a_.text(0.5, 0.5, "Load SRIM %s files (left)" % kind, ha="center", va="center",
                        transform=a_.transAxes, color="#888")
        self.tcanvas.draw_idle()

    def on_tab(self, _e=None):
        """Data treatment opened with no group: the one of Import data, if it has _nobg files."""
        if self.nb.index(self.nb.select()) == 1 and not self.tfiles and self.v_name.get().strip():
            self.t_dir.set(self.v_dir.get().strip())
            self.t_name.set(self.v_name.get().strip())
            self.t_load(quiet=True)

    def t_select(self):
        p = filedialog.askopenfilename(title="A file without background", initialdir=self.t_dir.get() or
                                       self.v_dir.get() or None,
                                       filetypes=[("Without background", "*_nobg.txt"), ("All files", "*.*")])
        if not p:
            return
        m = NOBG_RE.match(os.path.basename(p))
        if not m:
            messagebox.showerror("Select file", "The name does not end in _ionospectra<n>_nobg.txt:\n"
                                 + os.path.basename(p))
            return
        self.t_dir.set(os.path.normpath(os.path.dirname(p)))
        self.t_name.set(m.group("name"))
        self.t_load()

    def t_load(self, quiet=False):
        folder, name = self.t_dir.get().strip(), self.t_name.get().strip()
        if not os.path.isdir(folder) or not name:
            if not quiet:
                messagebox.showerror("Load", "Choose an existing folder and the NAME of the data group.")
            return
        found = []
        for f in os.listdir(folder):
            m = NOBG_RE.match(f)
            if m and m.group("name") == name:
                found.append((int(m.group("n")), f))
        files, bad = [], []
        for n, f in sorted(found):
            try:
                d = read_nobg(os.path.join(folder, f))
            except Exception as e:
                bad.append("%s (%s)" % (f, e))
                continue
            d.update(file=f, n=n, elapsed=info_number(d["info"], "Elapsed Time"))
            files.append(d)
        # fluence at the end of each spectrum: "Accumulated Fluence" of line 1 or, in older files,
        # the sum of the "Fluence" of the files listed up to this one
        acc = 0.0
        for d in files:
            f = info_number(d["info"], "Fluence")
            acc = acc + f if f is not None and acc is not None else None
            a = info_number(d["info"], "Accumulated Fluence")
            d["fluence"] = a if a is not None else acc
            d["flu_src"] = "Accumulated Fluence" if a is not None else "sum of Fluence"
        sel = set(self.ttv.selection())
        self.tfiles = files
        self.ttv.delete(*self.ttv.get_children())
        for i, d in enumerate(files):
            self.ttv.insert("", "end", iid=str(i), values=(
                i + 1, d["file"], "%g" % d["elapsed"] if d["elapsed"] is not None else "—",
                "%.4g" % d["fluence"] if d["fluence"] is not None else "—", ""),
                tags=("disabled",) if d["file"] in self.t_disabled else ())
        # spectra types: the columns after the wavelength (the ones of the first file)
        names = files[0]["cols"][1:] if files else []
        self.cb_tcol.configure(values=names)
        if names and self.t_col.get() not in names:
            self.t_col.set(names[-1])
        if files and not (self.t_x1.get().strip() and self.t_x2.get().strip()):
            self.t_roi_all()
        keep = [i for i in sel if i in self.ttv.get_children()]
        self.ttv.selection_set(keep or (("0",) if files else ()))
        txt = "%d file(s)" % len(files)
        if files and any(d["fluence"] is None for d in files):
            txt += "; %d without fluence (no beam area when they were written)" % sum(
                d["fluence"] is None for d in files)
        if files and files[0]["flu_src"] != "Accumulated Fluence":
            txt += "\nfluence = sum of the Fluence of the files listed (no Accumulated Fluence in line 1)"
        if bad:
            txt += "\nnot read: " + ", ".join(bad)
        self.l_tmsg.configure(text=txt if found else "No file %s_ionospectra<n>_nobg.txt in this folder" % name,
                              foreground="#666" if found and not bad else "#c00")
        self.t_selected()

    def t_enable(self, on, every=False):
        """Disable / enable the selected spectra (every: enable all of them); the integrated area vs
        fluence is drawn again without / with them."""
        items = self.tfiles if every else self.t_sel()
        if not items:
            return
        for d in items:
            (self.t_disabled.discard if on else self.t_disabled.add)(d["file"])
        for i, d in enumerate(self.tfiles):
            self.ttv.item(str(i), tags=("disabled",) if d["file"] in self.t_disabled else ())
        n = sum(d["file"] in self.t_disabled for d in self.tfiles)
        self.l_tmsg.configure(text="%d file(s), %d disabled" % (len(self.tfiles), n),
                              foreground="#c00" if n else "#666")
        self.t_draw()

    def t_enabled(self, d):
        return d["file"] not in self.t_disabled

    # --- files written by Save area vs fluence

    def t_saved_add(self, path, select=False):
        """Add (or read again, if already listed) a file of Save area vs fluence to the list."""
        p = os.path.normpath(path)
        try:
            d = read_area_file(p)
        except (OSError, ValueError) as e:
            self.l_tmsg.configure(text="Cannot read %s: %s" % (p, e), foreground="#c00")
            return None
        values = (os.path.basename(p), "%s  %s" % (d["info"].get("region of interest", ""),
                                                   d["info"].get("spectra", "").split("(")[0]))
        iid = next((i for i, x in self.t_saved.items() if os.path.normcase(x["path"]) == os.path.normcase(p)), None)
        if iid is None:
            iid = "s%d" % (max([int(i[1:]) for i in self.t_saved] or [0]) + 1)
            d["k"], d["c"] = 1.0, 0.0                   # shown area = area × k + c (Add / Multiply …)
            d["pin"] = False                            # Show ☑: always drawn
            self.stv.insert("", "end", iid=iid, values=("☐", "") + values + (self.t_op_text(d),))
        else:
            old = self.t_saved[iid]
            d["k"], d["c"], d["pin"] = old["k"], old["c"], old.get("pin", False)
            self.stv.item(iid, values=("☑" if d["pin"] else "☐", "") + values + (self.t_op_text(d),))
        self.t_saved[iid] = d
        if select:
            self.stv.selection_set((iid,))
            self.stv.see(iid)
        return iid

    @staticmethod
    def t_op_text(d):
        if d["k"] == 1 and d["c"] == 0:
            return ""
        return ("× %.4g" % d["k"] if d["k"] != 1 else "") + (" %s %.4g" % ("+" if d["c"] > 0 else "−", abs(d["c"]))
                                                            if d["c"] else "")

    def t_saved_op(self, op):
        """Add / subtract / multiply / divide the areas of the saved curves selected by the value
        (only in the plot; the files do not change), or reset them."""
        sel = self.stv.selection()
        if not sel:
            self.l_sops.configure(text="Select the curve(s) in the list below.", foreground="#c00")
            return
        if op != "reset":
            try:
                v = float(self.t_sval.get().replace(",", "."))
                if op == "÷" and v == 0:
                    raise ValueError
            except ValueError:
                self.l_sops.configure(text="Value: not a number%s." % (" (≠ 0)" if op == "÷" else ""),
                                      foreground="#c00")
                return
        for iid in sel:
            d = self.t_saved[iid]
            if op == "reset":
                d["k"], d["c"] = 1.0, 0.0
            elif op == "+":
                d["c"] += v
            elif op == "−":
                d["c"] -= v
            elif op == "×":
                d["k"], d["c"] = d["k"] * v, d["c"] * v
            else:
                d["k"], d["c"] = d["k"] / v, d["c"] / v
            self.stv.set(iid, "op", self.t_op_text(d))
        self.l_sops.configure(text="%s: %d curve(s)" % ("reset" if op == "reset" else "%s %g" % (op, v), len(sel)),
                              foreground="#070")
        self.t_draw()

    def t_show_panel(self, saved):
        """Below the list of the _nobg files: line 1 of the file, or the operations on the saved curves."""
        a, b = (self.t_fops, self.t_finfo) if saved else (self.t_finfo, self.t_fops)
        if not a.winfo_manager():
            b.pack_forget()
            a.pack(fill="both", expand=True)

    def t_pin_click(self, ev):
        """Click on the Show column: the curve is always drawn (☑) or only when selected (☐)."""
        if self.stv.identify_region(ev.x, ev.y) != "cell" or self.stv.identify_column(ev.x) != "#1":
            return None
        iid = self.stv.identify_row(ev.y)
        if not iid:
            return None
        self.t_set_pin(iid, not self.t_saved[iid].get("pin"))
        self.t_mode = "saved"
        self.t_draw()
        return "break"                          # the selection does not change

    def t_set_pin(self, iid, on):
        self.t_saved[iid]["pin"] = bool(on)
        self.stv.set(iid, "pin", "☑" if on else "☐")

    def t_shown_saved(self):
        """Saved curves drawn: the ones selected and the ones with Show ☑, in the order of the list."""
        sel = set(self.stv.selection())
        return [i for i in self.stv.get_children() if i in sel or self.t_saved[i].get("pin")]

    def t_saved_add_dialog(self):
        ps = filedialog.askopenfilenames(title="Area vs fluence files", initialdir=self.t_dir.get() or None,
                                         filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        for p in ps:
            self.t_saved_add(p)

    def t_saved_remove(self):
        for iid in self.stv.selection():
            self.stv.delete(iid)
            self.t_saved.pop(iid, None)
        self.t_mode = "files"
        self.t_draw()

    def t_saved_selected(self):
        if not self.stv.selection():
            return
        self.t_mode = "saved"
        self.t_draw()

    def t_selected(self):
        if self.ttv.selection():
            self.t_mode = "files"
            if self.stv.selection():
                self.stv.selection_set(())
        sel = self.t_sel()
        self.t_info.configure(state="normal")
        self.t_info.delete("1.0", "end")
        if len(sel) == 1:
            d = sel[0]
            self.t_info.insert("end", "\n".join("%s: %s" % (k, v) if k else v for k, v in d["header"]))
        elif sel:
            self.t_info.insert("end", "%d files selected (select one to see its line 1)" % len(sel))
        self.t_info.configure(state="disabled")
        self.t_draw()

    def t_sel(self):
        return [self.tfiles[int(i)] for i in sorted(self.ttv.selection(), key=int)] if self.tfiles else []

    def t_roi(self):
        """(low, high) of the region of interest (nm) or None."""
        try:
            a, b = float(self.t_x1.get()), float(self.t_x2.get())
        except ValueError:
            return None
        return (min(a, b), max(a, b))

    def t_roi_all(self):
        if self.tfiles:
            x = self.tfiles[0]["data"][:, 0]
            self.t_busy = True
            self.t_x1.set("%g" % x.min())
            self.t_x2.set("%g" % x.max())
            self.t_busy = False
            self.t_draw()

    def t_roi_typed(self):
        """Limits typed: the lines move there and the areas are calculated again (debounced)."""
        for e, v in zip(self.t_ents, (self.t_x1, self.t_x2)):
            try:
                float(v.get())
                e.configure(bg="white")
            except ValueError:
                e.configure(bg="#f4a6a6")
        if self.t_busy:
            return
        if self.t_job:
            self.root.after_cancel(self.t_job)
        self.t_job = self.root.after(400, self.t_roi_moved)

    def t_roi_moved(self):
        self.t_job = None
        roi = self.t_roi()
        if roi is None or len(self.t_lines) != 2:
            self.t_draw()
            return
        for ln, x in zip(self.t_lines, (float(self.t_x1.get()), float(self.t_x2.get()))):
            ln.set_xdata([x, x])
        self.t_draw_upper()
        self.tcanvas.draw_idle()

    def t_column(self, d):
        """Index of the column of the chosen spectra in file d, or None."""
        c = self.t_col.get()
        return d["cols"].index(c) if c in d["cols"] else None

    def t_areas(self):
        """Integrated area (∫ y dλ in the region of interest) of the chosen spectra of each file."""
        roi = self.t_roi()
        out = []
        for d in self.tfiles:
            j = self.t_column(d)
            if roi is None or j is None:
                out.append(None)
                continue
            x, y = d["data"][:, 0], d["data"][:, j]
            m = (x >= roi[0]) & (x <= roi[1])
            out.append(trapz(y[m], x[m]) if m.sum() > 1 else None)
        return out

    def t_colors(self, sel):
        n = len(sel)
        if n <= len(TAB10):
            return [TAB10[i] for i in range(n)]
        ks = [d["n"] for d in sel]
        lo, hi = min(ks), max(ks)
        return [cm.viridis((d["n"] - lo) / max(hi - lo, 1)) for d in sel]

    def t_draw(self):
        self.t_scolors = {}
        self._t_draw()
        self.t_paint_colours()

    def t_paint_colours(self):
        """Column Colour of the saved files list: a square of the colour of each curve drawn."""
        for w in self.t_swatches:
            w.destroy()
        self.t_swatches = []
        for iid, c in self.t_scolors.items():
            if not self.stv.exists(iid):
                continue
            b = self.stv.bbox(iid, "colour")
            if not b:                          # row not visible (scrolled)
                continue
            x, y, w, h = b
            sw = max(min(w - 12, 26), 6)
            sq = tk.Frame(self.stv, bg=c, width=sw, height=max(h - 8, 6), highlightthickness=1,
                          highlightbackground="#444")
            sq.place(x=x + (w - sw) // 2, y=y + 4)
            sq.bind("<Button-1>", lambda e, i=iid: self.stv.selection_set((i,)))
            self.t_swatches.append(sq)

    def _t_draw(self):
        f = self.tfig
        f.clear()
        if getattr(self, "t_sub", "data") == "models":
            self.m_draw()
            return
        saved = self.t_mode == "saved" and bool(self.t_shown_saved())
        self.t_show_panel(saved)
        if saved:
            self.t_draw_saved()
            return
        gs = f.add_gridspec(2, 1, height_ratios=[1, 1])
        self.t_axu = f.add_subplot(gs[0])
        ax = self.t_axl = f.add_subplot(gs[1])
        sel = self.t_sel()
        colors = self.t_colors(sel)
        col = self.t_col.get()
        for d, c in zip(sel, colors):
            j = self.t_column(d)
            if j is not None:
                ax.plot(d["data"][:, 0], d["data"][:, j], color=c, lw=1, label="%s" % d["n"])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(col or "Intensity", fontsize=8)
        if self.t_log.get():
            ax.set_yscale("log", nonpositive="mask")
        ax.grid(alpha=0.3)
        if sel and len(sel) <= 12:
            ax.legend(fontsize=7, title="spectrum n", title_fontsize=7, loc="upper right")
        if not self.tfiles:
            ax.text(0.5, 0.5, "Select a _nobg.txt file (left)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
        self.t_lines = []
        roi = self.t_roi()
        if roi is not None and self.tfiles:
            for v in (self.t_x1, self.t_x2):
                self.t_lines.append(ax.axvline(float(v.get()), color="#d62728", lw=1.3, ls="--", picker=True))
        # green cursor: a vertical line dragged with the mouse, with its x written in the plot
        self.t_cline = self.t_ctext = None
        if self.tfiles:
            x = self.tfiles[0]["data"][:, 0]
            if self.t_cur_x is None or not x.min() <= self.t_cur_x <= x.max():
                self.t_cur_x = float((x.min() + x.max()) / 2)
            self.t_cline = ax.axvline(self.t_cur_x, color="#2b9348", lw=1.4)
            self.t_ctext = ax.text(self.t_cur_x, 0.98, " x = %.4g nm" % self.t_cur_x, transform=ax.get_xaxis_transform(),
                                   color="#2b9348", fontsize=8, va="top", ha="left", fontweight="bold")
        self.t_draw_upper()
        self.tcanvas.draw_idle()

    def t_draw_upper(self):
        """Integrated area of the region of interest vs the accumulated fluence (all the files; the
        selected ones with the colours of the lower plot)."""
        ax = self.t_axu
        ax.clear()
        areas = self.t_areas()
        for i, a in enumerate(areas):
            self.ttv.set(str(i), "a", "%.5g" % a if a is not None else "—")
        pts = [(d["fluence"], a, d) for d, a in zip(self.tfiles, areas)
               if d["fluence"] is not None and a is not None and self.t_enabled(d)]
        ndis = sum(not self.t_enabled(d) for d in self.tfiles)
        if ndis:
            ax.text(0.99, 0.97, "%d disabled spectra not used" % ndis, ha="right", va="top", fontsize=8,
                    color="#d00000", transform=ax.transAxes)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#888", ms=3, lw=0.8)
            sel = self.t_sel()
            for d, c in zip(sel, self.t_colors(sel)):
                for fl, a, dd in pts:
                    if dd is d:
                        ax.plot(fl, a, "o", color=c, ms=7, mec="k", mew=0.5)
        roi = self.t_roi()
        ax.set_title("Integrated area %s" % ("%g – %g nm" % roi if roi else "(no region of interest)"), fontsize=9)
        ax.set_xlabel("Fluence (ions/cm²)")
        ax.set_ylabel("Area (%s × nm)" % (self.t_col.get().split("(")[0] or "y"), fontsize=8)
        ax.grid(alpha=0.3)
        if self.tfiles and not pts:
            ax.text(0.5, 0.5, "No fluence in the files (or no region of interest)", ha="center", va="center",
                    transform=ax.transAxes, color="#c00")

    def t_curve_y(self, d):
        """Area of a saved curve as drawn: / its maximum if chosen, then × k + c (Add / Multiply …,
        so the operations also act on the normalized curve)."""
        y = d["area"]
        if self.t_snorm.get() and len(y):
            m = np.nanmax(y)
            y = y / m if m else y
        return y * d["k"] + d["c"]

    def t_draw_saved(self):
        """Only the upper plot: the area vs fluence of the saved files selected in their list."""
        ax = self.tfig.add_subplot(111)
        self.t_axl, self.t_lines = None, []
        sel = self.t_shown_saved()
        norm = self.t_snorm.get()
        for i, iid in enumerate(sel):
            d = self.t_saved[iid]
            c = self.t_scolors[iid] = TAB10[i % len(TAB10)]
            y = self.t_curve_y(d)
            op = self.t_op_text(d)
            ax.plot(d["fluence"], y, "o-", color=c, ms=4, lw=1,
                    label="%s  (%s)%s" % (os.path.basename(d["path"]), d["info"].get("region of interest", ""),
                                          "  " + op if op else ""))
        cols = {self.t_saved[i]["info"].get("spectra", "") for i in sel}
        ax.set_title("Saved integrated area vs fluence" + ("  (each curve / its maximum)" if norm else ""),
                     fontsize=9)
        ax.set_xlabel("Fluence (ions/cm²)")
        ax.set_ylabel("Area / maximum" if norm else
                      "Area (%s × nm)" % (cols.pop().split("(")[0] if len(cols) == 1 else "y"), fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        self.tcanvas.draw_idle()

    def t_press(self, ev):
        """Start dragging a vertical line of the region of interest (not with zoom / pan)."""
        if ev.inaxes is None or ev.inaxes is not getattr(self, "t_axl", None) or self.ttb.mode or ev.button != 1:
            return
        best = None
        lines = list(enumerate(self.t_lines)) + ([("cur", self.t_cline)] if self.t_cline is not None else [])
        for i, ln in lines:
            px = self.t_axl.transData.transform((ln.get_xdata()[0], 0))[0]
            if abs(px - ev.x) < 8 and (best is None or abs(px - ev.x) < best[1]):
                best = (i, abs(px - ev.x))
        self.t_drag = best[0] if best else None

    def t_motion(self, ev):
        if self.t_drag is None or ev.inaxes is not self.t_axl or ev.xdata is None:
            return
        if self.t_drag == "cur":                # green cursor: only its x
            self.t_cur_x = float(ev.xdata)
            self.t_cline.set_xdata([ev.xdata, ev.xdata])
            self.t_ctext.set_x(ev.xdata)
            self.t_ctext.set_text(" x = %.4g nm" % ev.xdata)
            self.tcanvas.draw_idle()
            return
        self.t_lines[self.t_drag].set_xdata([ev.xdata, ev.xdata])
        self.t_busy = True
        (self.t_x1, self.t_x2)[self.t_drag].set("%.4g" % ev.xdata)
        self.t_busy = False
        self.t_draw_upper()
        self.tcanvas.draw_idle()

    def t_save(self):
        roi = self.t_roi()
        if not self.tfiles or roi is None:
            messagebox.showerror("Save", "Load the files and define the region of interest.")
            return
        p = filedialog.asksaveasfilename(
            title="Save integrated area vs fluence", defaultextension=".txt", initialdir=self.t_dir.get() or None,
            initialfile="%s_area_%g-%gnm.txt" % (self.t_name.get().strip() or "ionolum", roi[0], roi[1]),
            filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not p:
            return
        areas = self.t_areas()
        try:
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("# Integrated area vs fluence (ionolum)\n# folder: %s\n# name: %s\n" % (
                    self.t_dir.get().strip(), self.t_name.get().strip()))
                fh.write("# spectra: %s\n# region of interest: %g - %g nm\n" % (self.t_col.get(), roi[0], roi[1]))
                fh.write("# fluence: %s\n" % (self.tfiles[0]["flu_src"]))
                dis = [d["file"] for d in self.tfiles if not self.t_enabled(d)]
                fh.write("# disabled (not included): %s\n" % (", ".join(dis) if dis else "none"))
                fh.write("Fluence(ions/cm2)\tArea\tElapsed_time(s)\tn\tFile\n")
                for d, a in zip(self.tfiles, areas):
                    if not self.t_enabled(d):
                        continue
                    fh.write("%s\t%s\t%s\t%d\t%s\n" % (
                        "%.6g" % d["fluence"] if d["fluence"] is not None else "nan",
                        "%.6g" % a if a is not None else "nan",
                        "%g" % d["elapsed"] if d["elapsed"] is not None else "nan", d["n"], d["file"]))
        except OSError as e:
            messagebox.showerror("Save", str(e))
            return
        self.l_tmsg.configure(text="Saved: %s" % p, foreground="#070")
        self.t_saved_add(p)

    # ------------------------------------------------ experimental setup
    def build_setup(self, tab):
        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, width=430)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)
        self.sv = {}

        def var(key, value):
            v = (tk.BooleanVar if isinstance(value, bool) else tk.StringVar)(value=value)
            self.sv[key] = v
            return v

        def row(parent, r, text, key, value, unit="", width=10):
            ttk.Label(parent, text=text).grid(row=r, column=0, sticky="w", padx=6, pady=2)
            e = ttk.Entry(parent, textvariable=var(key, value), width=width)
            e.grid(row=r, column=1, sticky="w", pady=2)
            ttk.Label(parent, text=unit).grid(row=r, column=2, sticky="w", padx=4)
            return e

        # ---- experiment
        g = ttk.LabelFrame(left, text="Experiment")
        g.pack(fill="x", padx=6, pady=6)
        var("mode", "external")
        ttk.Radiobutton(g, text="External beam (in air / gas)", value="external", variable=self.sv["mode"]
                        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 0))
        ttk.Radiobutton(g, text="Internal beam (vacuum chamber)", value="internal", variable=self.sv["mode"]
                        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))

        # ---- beam
        b = ttk.LabelFrame(left, text="Beam")
        b.pack(fill="x", padx=6, pady=(0, 6))
        row(b, 0, "Element", "ion", "H", width=6)
        ttk.Label(b, text="Mass (u)").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        self.cb_mass = ttk.Combobox(b, textvariable=var("mass", ""), width=24)
        self.cb_mass.grid(row=1, column=1, columnspan=2, sticky="w", pady=2)
        self.l_ion = ttk.Label(b, text="", foreground="#666")
        self.l_ion.grid(row=0, column=2, sticky="w", padx=4)
        row(b, 2, "Energy", "energy", "2000", "keV")
        row(b, 3, "Diameter", "diam", "1", "mm")
        row(b, 4, "Area", "area", "", "mm²  (circular section)")
        row(b, 5, "Charge state", "q", "1", "(ions = Q / (q·e), for the fluence)", width=6)
        self.last_ion = None

        # ---- external beam
        x = self.f_ext = ttk.LabelFrame(left, text="External beam")
        x.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(x, text="Atmosphere").grid(row=0, column=0, sticky="nw", padx=6, pady=2)
        fa = ttk.Frame(x)
        fa.grid(row=0, column=1, columnspan=2, sticky="w")
        var("atm", "Air")
        for k in ATMOSPHERES:
            ttk.Radiobutton(fa, text=k, value=k, variable=self.sv["atm"]).pack(side="left", padx=(0, 8))
        row(x, 1, "Pressure", "p", "1013.25", "mbar")
        row(x, 2, "Temperature", "t", "20", "°C")
        self.l_gas = ttk.Label(x, text="", foreground="#666")
        self.l_gas.grid(row=3, column=0, columnspan=3, sticky="w", padx=6)
        row(x, 4, "Distance nozzle – sample", "dist", "5", "mm")
        ttk.Checkbutton(x, text="Exit window at the nozzle", variable=var("win", False)
                        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 0))
        row(x, 6, "   composition", "wcomp", "Si3N4", "(formula or 'Si 3 N 4')")
        row(x, 7, "   thickness", "wthick", "100", "nm")
        row(x, 8, "   density", "wrho", "3.44", "g/cm³")
        self.ext_widgets = [w for w in x.winfo_children() + fa.winfo_children() if w is not fa]

        # ---- TRIM
        t = ttk.LabelFrame(left, text="TRIM (SRIM) calculation of the beam at the sample")
        t.pack(fill="x", padx=6, pady=(0, 6))
        # asked the first time TRIM would run (setup_calc); off: energy and diameter of the beam
        ttk.Checkbutton(t, text="Consider the energy loss and the spreading of the beam (external beam)",
                        variable=var("trim", True)).grid(row=0, column=0, columnspan=3, sticky="w", padx=6,
                                                         pady=(4, 0))
        self.trim_asked = self.trim_asking = False
        row(t, 1, "Number of ions", "nions", "200", "(~5 s per 100 ions in 10 mm of air)")
        ttk.Label(t, text="SRIM folder").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(t, textvariable=var("srim", SRIM_DIR), width=34).grid(row=2, column=1, columnspan=2,
                                                                      sticky="we", pady=2, padx=(0, 6))
        ttk.Label(t, text="TRIM.exe of this folder (TRIM.IN and SRIM Outputs are overwritten)",
                  foreground="#666", wraplength=400).grid(row=3, column=0, columnspan=3, sticky="w", padx=6)
        fb = ttk.Frame(t)
        fb.grid(row=4, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(fb, text="Recalculate when a parameter changes", variable=var("auto", True)
                        ).pack(side="left")
        ttk.Button(fb, text="Calculate now", command=lambda: self.setup_calc(force=True)).pack(side="left", padx=6)
        self.b_stop = ttk.Button(fb, text="Stop TRIM", command=self.setup_stop, state="disabled")
        self.b_stop.pack(side="left")
        self.l_calc = ttk.Label(t, text="", wraplength=400, justify="left")
        self.l_calc.grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))

        # ---- results
        r = ttk.LabelFrame(left, text="Beam at the sample")
        r.pack(fill="x", padx=6, pady=(0, 6))
        self.res = {}
        for i, (k, lab) in enumerate((("E", "Energy"), ("dE", "Energy loss"), ("A", "Area"),
                                      ("sig", "Lateral spread"), ("ang", "Angular spread"), ("frac", "Ions reaching it"))):
            ttk.Label(r, text=lab + ":").grid(row=i, column=0, sticky="w", padx=6, pady=1)
            self.res[k] = ttk.Label(r, text="—", font=("TkDefaultFont", 9, "bold") if k in ("E", "A") else None)
            self.res[k].grid(row=i, column=1, sticky="w", padx=6, pady=1)

        self.fig_s = Figure(figsize=(8, 5), dpi=100, layout="constrained")
        self.canvas_s = FigureCanvasTkAgg(self.fig_s, master=right)
        tb = NavigationToolbar2Tk(self.canvas_s, right, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        self.canvas_s.get_tk_widget().pack(fill="both", expand=True)

        self.calc = {"thread": None, "proc": [], "pending": False, "job": None, "cache": {}, "key": None,
                     "out": None, "t0": 0, "stopped": False}
        self.beam = None                       # beam at the sample (dict) used by the other tabs
        self.link_busy = False
        for k, v in self.sv.items():
            v.trace_add("write", lambda *a, k=k: self.setup_changed(k))
        self.setup_changed("ion")

    def setup_changed(self, key):
        if self.link_busy:
            return
        sv = self.sv
        self.link_busy = True
        try:
            if key == "ion":
                self.update_ion()
            elif key in ("diam", "area"):      # diameter <-> area of the circular section
                try:
                    v = float(sv[key].get())
                    if key == "diam":
                        sv["area"].set("%.4g" % (np.pi * v * v / 4))
                    else:
                        sv["diam"].set("%.4g" % np.sqrt(4 * v / np.pi))
                except ValueError:
                    pass
            if not sv["area"].get().strip():
                try:
                    sv["area"].set("%.4g" % (np.pi * float(sv["diam"].get()) ** 2 / 4))
                except ValueError:
                    pass
        finally:
            self.link_busy = False
        ext = sv["mode"].get() == "external"
        for w in self.ext_widgets:
            try:
                w.configure(state="normal" if ext else "disabled")
            except tk.TclError:
                pass
        try:
            self.l_gas.configure(text="density %.4g g/cm³" % gas_density(sv["atm"].get(), float(sv["p"].get()),
                                                                           float(sv["t"].get())))
        except (ValueError, KeyError, ZeroDivisionError):
            self.l_gas.configure(text="")
        if self.calc["job"]:
            self.root.after_cancel(self.calc["job"])
        self.calc["job"] = self.root.after(800, self.setup_calc)

    def update_ion(self):
        from ionolum_lib.srim_tools import isotope_masses
        s = self.sv["ion"].get().strip()
        isos = isotope_masses(s) if s else []
        if not isos:
            self.cb_mass.configure(values=())
            self.l_ion.configure(text="not an element", foreground="#c00")
            self.last_ion = None
            return
        sym = s[:1].upper() + s[1:].lower()
        vals = ["%.5f  (%d%s, %.4g %%)" % (m, a, sym, ab) for a, m, ab in isos]
        self.cb_mass.configure(values=vals)
        if sym != self.last_ion:            # new element: its most abundant isotope
            self.sv["mass"].set(vals[0])
        self.last_ion = sym
        self.l_ion.configure(text="", foreground="#666")

    def setup_params(self):
        """Parameters of the setup (dict) or ValueError with a message."""
        sv = self.sv
        p = {"mode": sv["mode"].get()}

        def num(key, name, low=None, strict=False):
            try:
                v = float(sv[key].get())
            except ValueError:
                raise ValueError("%s: not a number" % name)
            if low is not None and (v <= low if strict else v < low):
                raise ValueError("%s must be %s %g" % (name, ">" if strict else "≥", low))
            return v

        if self.last_ion is None:
            raise ValueError("Beam element: not an element of the periodic table")
        p["ion"] = self.last_ion
        try:
            p["mass"] = float(sv["mass"].get().split()[0])
        except (ValueError, IndexError):
            raise ValueError("Mass of the ion: not a number")
        p["energy"] = num("energy", "Energy", 0, True)
        p["diam"] = num("diam", "Diameter", 0, True)
        if p["mode"] == "external":
            p["atm"] = sv["atm"].get()
            p["rho"] = gas_density(p["atm"], num("p", "Pressure", 0, True), num("t", "Temperature", -273.15, True))
            p["dist"] = num("dist", "Distance", 0)
            p["win"] = None
            if sv["win"].get():
                from ionolum_lib.srim_tools import srim_parse_comp
                try:
                    comp = srim_parse_comp(sv["wcomp"].get())
                except ValueError as e:
                    raise ValueError("Exit window composition: %s" % e)
                p["win"] = (sv["wcomp"].get().strip(), tuple(sorted(comp.items())),
                            num("wthick", "Window thickness", 0, True), num("wrho", "Window density", 0, True))
            p["nions"] = int(num("nions", "Number of ions", 1))
            p["srim"] = sv["srim"].get().strip()
        return p

    @staticmethod
    def setup_key(p):
        """Parameters that decide the TRIM result of an external beam (key of the cache)."""
        return (p["ion"], p["mass"], p["energy"], p["atm"], round(p["rho"], 9), p["dist"], p["win"], p["nions"])

    def setup_calc(self, force=False):
        """Beam at the sample: internal beam = the beam; external = TRIM through the window (if any)
        and the atmosphere (in a thread). Called after a change (debounced) or by Calculate now."""
        c = self.calc
        c["job"] = None
        if self.trim_asking:                    # the question is open: later
            c["job"] = self.root.after(500, lambda: self.setup_calc(force))
            return
        try:
            p = self.setup_params()
        except ValueError as e:
            self.l_calc.configure(text=str(e), foreground="#c00")
            self.clear_beam(str(e))
            return
        external = p["mode"] == "external" and (p["dist"] or p["win"])
        if external and not self.trim_asked:   # the first time TRIM would be needed: ask
            self.trim_asked = self.trim_asking = True
            try:
                ans = messagebox.askyesno(
                    "Experimental setup", "External beam: consider the energy loss and the spreading of the "
                    "beam between the nozzle and the sample?\n\nYes: they are calculated with SRIM (TRIM) "
                    "whenever needed.\nNo: the energy and the diameter at the sample are the ones of the beam "
                    "(they do not change with the distance).\n\n(It can be changed later in Experimental "
                    "setup › TRIM.)", parent=self.root)
            finally:
                self.trim_asking = False
            self.sv["trim"].set(ans)          # (its trace schedules setup_calc again: harmless)
        if not external or not self.sv["trim"].get():
            a = np.pi * p["diam"] ** 2 / 4
            self.show_beam(p, {"E": p["energy"], "E_sd": 0.0, "A0": a, "A": a, "d": p["diam"], "sigma": 0.0,
                               "angle": 0.0, "frac": 1.0, "e": None, "yz": None, "pts": None})
            self.l_calc.configure(text="Internal beam: no energy loss nor spreading before the sample."
                                  if p["mode"] == "internal" else
                                  "Energy loss and spreading not considered: energy and diameter at the sample "
                                  "= the ones of the beam." if external else "No gas nor window: the beam itself.",
                                  foreground="#333")
            return
        key = self.setup_key(p)
        if key in c["cache"] and not force:
            try:
                self.show_beam(p, beam_at_sample(c["cache"][key], p["nions"], p["diam"]))
                self.l_calc.configure(text="TRIM result of %d ions (already calculated)." % p["nions"],
                                      foreground="#333")
            except RuntimeError as e:
                self.l_calc.configure(text="✖ %s" % e, foreground="#c00")
                self.clear_beam(str(e))
            return
        if not self.sv["auto"].get() and not force:
            self.l_calc.configure(text="Parameters changed: press Calculate now.", foreground="#b60")
            self.clear_beam("parameters changed: press Calculate now")
            return
        if getattr(self, "hooks", {}).get("srim_busy"):     # TRIM of the SRIM tab in the same folder
            self.l_calc.configure(text="TRIM of the SRIM tab is running: waiting…", foreground="#b60")
            c["job"] = self.root.after(3000, lambda: self.setup_calc(force))
            return
        if c["thread"] is not None:          # running: stop it and start again with the new values
            c["pending"] = True
            self.setup_stop(restart=True)
            return
        layers = []
        if p["win"]:
            name, comp, th, rho = p["win"]
            layers.append(("Window " + name, dict(comp), rho, th * 10.0, False))
        if p["dist"]:
            layers.append((p["atm"].split()[0], {k: v / sum(ATMOSPHERES[p["atm"]][0].values())
                                                 for k, v in ATMOSPHERES[p["atm"]][0].items()},
                           p["rho"], p["dist"] * 1e7, True))
        c.update(proc=[], out=None, key=key, t0=time.time(), stopped=False, pending=False, params=p)

        def work():
            try:
                d = trim_run_dir(p["srim"])
                c["out"] = run_trim_transmit(p["ion"], p["mass"], p["energy"], layers, p["nions"], d, c["proc"])
            except Exception as e:
                c["out"] = e
        c["thread"] = threading.Thread(target=work, daemon=True)
        c["thread"].start()
        self.clear_beam("TRIM running…")
        self.b_stop.configure(state="normal")
        self.setup_poll()

    def setup_poll(self):
        c = self.calc
        if c["thread"].is_alive():
            self.l_calc.configure(text="⏳ TRIM running (%d ions)… %.0f s" % (c["params"]["nions"],
                                                                          time.time() - c["t0"]),
                                  foreground="#05c")
            self.root.after(300, self.setup_poll)
            return
        c["thread"] = None
        self.b_stop.configure(state="disabled")
        out, p = c["out"], c["params"]
        if c["pending"]:                    # parameters changed during the run
            c["pending"] = False
            self.setup_calc(force=True)
            return
        if c["stopped"]:
            self.l_calc.configure(text="✖ TRIM stopped.", foreground="#c00")
            self.clear_beam("TRIM stopped")
        elif isinstance(out, Exception):
            self.l_calc.configure(text="✖ %s" % out, foreground="#c00")
            self.clear_beam(str(out))
        else:
            c["cache"][c["key"]] = out
            try:
                self.show_beam(p, beam_at_sample(out, p["nions"], p["diam"]))
                self.l_calc.configure(text="✔ TRIM: %d ions, %.0f s." % (p["nions"], time.time() - c["t0"]),
                                      foreground="#070")
            except RuntimeError as e:
                self.l_calc.configure(text="✖ %s" % e, foreground="#c00")
                self.clear_beam(str(e))

    def setup_stop(self, restart=False):
        c = self.calc
        if c["thread"] is None:
            return
        c["stopped"] = not restart
        for p in c["proc"]:
            try:
                p.kill()
            except OSError:
                pass

    def flu_info(self):
        """(fluence per coulomb, beam area at the sample mm², charge state) from the Experimental
        setup, or (None, reason)."""
        if getattr(self, "beam", None) is None:
            return None, "no beam area: see Experimental setup"
        try:
            q = int(self.sv["q"].get())
            if q < 1:
                raise ValueError
        except ValueError:
            return None, "charge state: not an integer ≥ 1 (Experimental setup)"
        a = self.beam["A"]
        return (fluence_per_coulomb(a, q), a, q), ""

    def refresh_fluence(self):
        """Column Φ of the list and the plots after a change of the beam area / charge state."""
        flu, why = self.flu_info()
        self.l_flu.configure(text="fluence: beam area at the sample %.4g mm², charge state %d" % (flu[1], flu[2])
                             if flu else why, foreground="#666" if flu else "#c00")
        for it in self.items:
            v = "%.4g" % (it["charge"] * flu[0] / FLUENCE_UNIT) if flu and it["charge"] else "—"
            self.tv.set(str(it["k"]), "f", v)
        if self.v_normk.get() == NORM_MODES[2] or self.v_low.get() == LOWER_PLOTS[2]:
            self.draw()

    def clear_beam(self, why):
        """No valid beam at the sample (running, error, parameters changed): nothing old shown."""
        self.beam = None
        for w in self.res.values():
            w.configure(text="—")
        self.fig_s.clear()
        self.fig_s.text(0.5, 0.5, why, ha="center", va="center", color="#888", wrap=True)
        self.canvas_s.draw_idle()
        if hasattr(self, "tv"):
            self.refresh_fluence()

    def show_beam(self, p, r):
        """Results of the beam at the sample (r: beam_at_sample) in the boxes and the plots."""
        self.beam = dict(r, params=p)
        self.refresh_fluence()
        e0 = p["energy"]
        set_ion = getattr(self, "hooks", {}).get("setion")
        if set_ion:                    # SRIM tab: the ion with the energy that reaches the sample
            set_ion(p["ion"], p["mass"], r["E"])
        self.res["E"].configure(text="%.1f keV" % r["E"] + ("   (± %.1f keV, FWHM %.1f keV)" % (r["E_sd"], 2.355 * r["E_sd"])
                                                            if r["E_sd"] else ""))
        self.res["dE"].configure(text="%.1f keV  (%.2f %%)" % (e0 - r["E"], 100 * (e0 - r["E"]) / e0))
        self.res["A"].configure(text="%.4g mm²  (diameter %.4g mm; at the nozzle %.4g mm², +%.2f %%)"
                                % (r["A"], r["d"], r["A0"], 100 * (r["A"] - r["A0"]) / r["A0"]))
        self.res["sig"].configure(text="σ = %.4g mm (per transverse axis)" % r["sigma"])
        self.res["ang"].configure(text="%.3g° (rms)" % r["angle"])
        self.res["frac"].configure(text="%.1f %%" % (100 * r["frac"]))
        f = self.fig_s
        f.clear()
        ax1, ax2 = f.subplots(1, 2)
        if r["e"] is not None:
            ax1.hist(r["e"], bins=max(10, min(60, len(r["e"]) // 5)), color="#2a78d6", alpha=0.8)
            ax1.axvline(r["E"], color="k", ls="--", lw=1)
        else:
            ax1.axvline(r["E"], color="#2a78d6", lw=2)
        ax1.set_xlabel("Energy at the sample (keV)")
        ax1.set_ylabel("Ions")
        ax1.set_title("%s %.0f keV → %.1f keV" % (p["ion"], e0, r["E"]), fontsize=10)
        ax1.grid(alpha=0.3)
        # beam spot: uniform disc at the nozzle + the deviations of the TRIM ions (beam_at_sample)
        pts = r.get("pts")
        if pts is None:
            pts = beam_spot(p["diam"])
        pts = pts[:6000]
        R0 = p["diam"] / 2
        ax2.plot(pts[:, 0], pts[:, 1], ".", ms=1, color="#2a78d6", alpha=0.4)
        t = np.linspace(0, 2 * np.pi, 200)
        ax2.plot(R0 * np.cos(t), R0 * np.sin(t), color="#555", ls="--", lw=1, label="at the nozzle (%.3g mm)" % p["diam"])
        ax2.plot(r["d"] / 2 * np.cos(t), r["d"] / 2 * np.sin(t), color="#d62728", lw=1.5,
                 label="at the sample (%.3g mm)" % r["d"])
        lim = max(r["d"] / 2 * 1.3, float(np.percentile(np.hypot(pts[:, 0], pts[:, 1]), 99.5)))
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_aspect("equal")
        ax2.set_xlabel("y (mm)")
        ax2.set_ylabel("z (mm)")
        ax2.set_title("Beam spot", fontsize=10)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(alpha=0.3)
        self.canvas_s.draw_idle()

    # ------------------------------------------------ session
    SESSION_OPTS = ("v_cur", "v_low", "v_normk", "v_log", "v_all", "v_art", "v_bgon", "v_bg")

    def save_session(self):
        """JSON with the data group (folder, name, files, selection), the options of the plot and the
        background, and the parameters of the Experimental setup (with the last TRIM result)."""
        p = filedialog.asksaveasfilename(
            title="Save session", defaultextension=".json", initialdir=self.v_dir.get() or None,
            initialfile=(self.v_name.get().strip() or "ionolum") + "_session.json",
            filetypes=[("Session (JSON)", "*.json"), ("All files", "*.*")])
        if not p:
            return
        s = {"program": "ionolum", "version": 1, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
             "import": {"folder": self.v_dir.get().strip(), "name": self.v_name.get().strip(),
                        "files": [{"spectrum": it["file"], "current": it["cfile"]} for it in self.items],
                        "selected": [int(i) for i in self.tv.selection()],
                        "options": {k[2:]: getattr(self, k).get() for k in self.SESSION_OPTS}},
             "setup": {k: v.get() for k, v in self.sv.items()},
             "treat": {"folder": self.t_dir.get().strip(), "name": self.t_name.get().strip(),
                       "spectra": self.t_col.get(), "roi": [self.t_x1.get(), self.t_x2.get()],
                       "selected": [int(i) for i in self.ttv.selection()], "log": self.t_log.get(),
                       "files": [d["file"] for d in self.tfiles],
                       "disabled": sorted(self.t_disabled),
                       "saved_area_files": [d["path"] for d in (self.t_saved[i] for i in self.stv.get_children())],
                       "saved_area_ops": [[d["k"], d["c"]] for d in (self.t_saved[i] for i in self.stv.get_children())],
                       "saved_area_show": [bool(self.t_saved[i].get("pin")) for i in self.stv.get_children()],
                       "saved_area_norm": self.t_snorm.get(), "cursor": self.t_cur_x},
             "models": {"preset": self.m_preset.get(), "expression": self.m_expr.get(), "x_unit": self.m_xu.get(),
                        "parameters": [{"value": r["v"].get(), "min": r["min"].get(), "max": r["max"].get(),
                                        "fit": r["fit"].get()} for r in self.m_par],
                        "fits": {self.t_saved[i]["path"]: list(self.m_fits[i]) for i in self.m_fits if i in self.t_saved},
                        "result": self.m_res.get("1.0", "end").strip(),
                        "srim_files": [self.m_srim[i]["path"] for i in self.mtv.get_children()],
                        "depth_unit": self.m_du.get()}}
        if self.hooks.get("srim_state"):        # SRIM tab: structure, ion, options, TRIM.IN
            try:
                s["srim"] = self.hooks["srim_state"]()
            except Exception as e:
                messagebox.showwarning("Save session", "The SRIM tab is not saved:\n%s" % e)
        b = self.beam
        if b is not None:
            s["setup_result"] = {k: b[k] for k in ("E", "E_sd", "A0", "A", "d", "sigma", "angle", "frac")}
            key = self.calc.get("key")
            if b.get("e") is not None and key in self.calc["cache"]:
                s["setup_result"]["transmit"] = self.calc["cache"][key].tolist()   # TRIM not run again
        try:
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(s, fh, indent=1, ensure_ascii=False)
        except OSError as e:
            messagebox.showerror("Save session", str(e))
            return
        self.l_bgmsg.configure(text="Session saved: %s" % p, foreground="#070")

    def import_session(self):
        p = filedialog.askopenfilename(title="Import session", initialdir=self.v_dir.get() or None,
                                       filetypes=[("Session (JSON)", "*.json"), ("All files", "*.*")])
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as fh:
                s = json.load(fh)
            imp, setup = s.get("import", {}), s.get("setup", {})
        except (OSError, ValueError, AttributeError) as e:
            messagebox.showerror("Import session", "Cannot read %s:\n%s" % (p, e))
            return
        # parameters of the setup first (a TRIM result saved goes to the cache: not run again)
        tr = (s.get("setup_result") or {}).get("transmit")
        if "trim" in setup:                  # the session says whether to consider them: not asked
            self.trim_asked = True
        for k, v in setup.items():
            if k in self.sv:
                try:
                    self.sv[k].set(v)
                except tk.TclError:
                    pass
        if tr:
            try:
                self.calc["cache"][self.setup_key(self.setup_params())] = np.array(tr, float).reshape(-1, 7)
            except (ValueError, KeyError):
                pass
        # data group: the folder saved or, if it no longer exists, the folder of the session file
        folder, name = imp.get("folder", ""), imp.get("name", "")
        if name and not os.path.isdir(folder):
            folder = os.path.dirname(p)
        notes = []
        if name:
            self.v_dir.set(os.path.normpath(folder))
            self.v_name.set(name)
            self.load()
            saved = [f.get("spectrum") for f in imp.get("files", [])]
            now = [it["file"] for it in self.items]
            if saved and saved != now:
                miss = [f for f in saved if f not in now]
                notes.append("%d spectra in the session, %d in the folder%s" % (
                    len(saved), len(now), " (%d missing, e.g. %s)" % (len(miss), miss[0]) if miss else ""))
        for k, v in imp.get("options", {}).items():
            if k == "norm":                 # sessions before the fluence: True = beam charge
                self.v_normk.set(NORM_MODES[1] if v else NORM_MODES[0])
            elif "v_" + k in self.SESSION_OPTS:
                getattr(self, "v_" + k).set(v)
        self.bg_label()
        sel = [str(i) for i in imp.get("selected", []) if 0 <= i < len(self.items)]
        if self.items:
            self.tv.selection_set(sel)
            if sel:
                self.tv.see(sel[0])
        self.draw()
        tr = s.get("treat") or {}
        if tr.get("name"):                  # Data treatment
            self.t_busy = True
            self.t_x1.set(tr.get("roi", ["", ""])[0])
            self.t_x2.set(tr.get("roi", ["", ""])[1])
            self.t_busy = False
            self.t_col.set(tr.get("spectra", ""))
            self.t_log.set(bool(tr.get("log")))
            self.t_cur_x = tr.get("cursor")
            self.t_dir.set(tr.get("folder") if os.path.isdir(tr.get("folder", "")) else os.path.dirname(p))
            self.t_name.set(tr["name"])
            self.t_disabled = set(tr.get("disabled", []))
            self.t_load(quiet=True)
            now = [d["file"] for d in self.tfiles]
            miss = [f for f in tr.get("files", []) if f not in now]
            if miss:
                notes.append("Data treatment: %d _nobg file(s) of the session not found (e.g. %s)"
                             % (len(miss), miss[0]))
            ts = [str(i) for i in tr.get("selected", []) if 0 <= i < len(self.tfiles)]
            if ts:
                self.ttv.selection_set(ts)
        ops = tr.get("saved_area_ops", [])
        for j, sp in enumerate(tr.get("saved_area_files", [])):
            if not os.path.exists(sp):
                alt = os.path.join(os.path.dirname(p), os.path.basename(sp))
                sp = alt if os.path.exists(alt) else sp
            iid = self.t_saved_add(sp)
            if iid is None:
                notes.append("saved area file not found: %s" % sp)
            elif j < len(ops):
                d = self.t_saved[iid]
                d["k"], d["c"] = float(ops[j][0]), float(ops[j][1])
                self.stv.set(iid, "op", self.t_op_text(d))
            if iid is not None and j < len(tr.get("saved_area_show", [])):
                self.t_set_pin(iid, tr["saved_area_show"][j])
        if any(d.get("pin") for d in self.t_saved.values()):
            self.t_mode = "saved"
        self.t_snorm.set(bool(tr.get("saved_area_norm")))
        if s.get("srim") and self.hooks.get("srim_restore"):
            try:
                self.hooks["srim_restore"](s["srim"])
            except Exception as e:
                notes.append("SRIM tab not restored: %s" % e)
        md = s.get("models") or {}
        if md:                              # Data treatment › Models
            self.m_expr.set(md.get("expression", ""))
            self.m_preset.set(md.get("preset", "Custom"))
            if md.get("x_unit") in self.X_UNITS:
                self.m_xu.set(md["x_unit"])
            for row, v in zip(self.m_par, md.get("parameters", [])):
                row["v"].set(v.get("value", "1"))
                row["min"].set(v.get("min", ""))
                row["max"].set(v.get("max", ""))
                row["fit"].set(bool(v.get("fit", True)))
            self.m_fits = {}
            for path, fit in (md.get("fits") or {}).items():
                iid = next((i for i, d in self.t_saved.items()
                            if os.path.basename(d["path"]) == os.path.basename(path)), None)
                if iid and len(fit) == 3:
                    self.m_fits[iid] = tuple(fit)
            self.m_write(md.get("result", ""))
            if md.get("depth_unit") in self.DEPTH_UNITS:
                self.m_du.set(md["depth_unit"])
            for i in list(self.m_srim):
                self.mtv.delete(i)
                self.m_srim.pop(i)
            for sp in md.get("srim_files", []):
                if not os.path.exists(sp):
                    alt = os.path.join(os.path.dirname(p), os.path.basename(sp))
                    sp = alt if os.path.exists(alt) else sp
                err = self.m_srim_add(sp) if os.path.exists(sp) else "SRIM file not found: %s" % sp
                if err:
                    notes.append(err)
        self.t_draw()
        self.l_bgmsg.configure(text="Session imported: %s%s" % (p, "".join("\n⚠ " + n for n in notes)),
                               foreground="#b60" if notes else "#070")

    # ------------------------------------------------ import
    def select_file(self):
        p = filedialog.askopenfilename(
            title="A file of the data group", initialdir=self.v_dir.get() or None,
            filetypes=[("Ionoluminescence files", "*_ionospectra*.txt *_current_K6485_esp_*.txt"),
                       ("Text files", "*.txt"), ("All files", "*.*")])
        if not p:
            return
        s = split_name(os.path.basename(p))
        if not s:
            messagebox.showerror("Select file", "The name does not end in _ionospectra<n>.txt or "
                                                "_current_K6485_esp_<n>.txt:\n" + os.path.basename(p))
            return
        self.v_dir.set(os.path.normpath(os.path.dirname(p)))
        self.v_name.set(s[0])
        self.load()

    def load(self):
        folder, name = self.v_dir.get().strip(), self.v_name.get().strip()
        if not os.path.isdir(folder) or not name:
            messagebox.showerror("Load", "Choose an existing folder and the NAME of the data group.")
            return
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            items, ncurr = load_group(folder, name)
        except Exception as e:
            messagebox.showerror("Load", "Error reading the files:\n%s" % e)
            return
        finally:
            self.root.configure(cursor="")
        if not items:
            messagebox.showwarning("Load", "No file %s_ionospectra<n>.txt in\n%s" % (name, folder))
        self.items = items
        self.show_info(ncurr)
        self.tv.delete(*self.tv.get_children())
        for it in items:
            c = it["current"]
            im = "%.3f" % (np.mean(c) * 1e9) if c is not None and len(c) else "—"
            q = "%.3f" % (it["charge"] * 1e9) if it["charge"] else "—"
            self.tv.insert("", "end", iid=str(it["k"]), values=(it["k"] + 1, it["file"], "%g" % it["t0"], im, q, "—"))
        self.refresh_fluence()
        self.sp_bg.configure(to=max(len(items), 1))
        self.v_bg.set(str(len(items)) if items else "")       # background: the last spectrum by default
        self.l_bgmsg.configure(text="")
        self.bg_label()
        if items:
            self.tv.selection_set(("0",))
        self.draw()

    # ------------------------------------------------ background
    def bg_item(self):
        """Spectrum chosen as background (field "spectrum #", 1 … number of spectra), or None."""
        try:
            k = int(self.v_bg.get()) - 1
        except ValueError:
            return None
        return self.items[k] if 0 <= k < len(self.items) else None

    def bg_label(self):
        bg = self.bg_item()
        ok = bg is not None or not self.items
        self.sp_bg.configure(bg="white" if ok else "#f4a6a6")
        self.l_bg.configure(text=bg["file"] if bg else ("" if ok else "no spectrum with this number (1 – %d)"
                                                          % len(self.items)),
                            foreground="#666" if ok else "#c00")

    def bg_changed(self, delay=0):
        if self.bg_job:
            self.root.after_cancel(self.bg_job)
        self.bg_job = self.root.after(delay, self._bg_changed)

    def _bg_changed(self):
        self.bg_job = None
        self.bg_label()
        if self.v_bgon.get():
            self.draw()

    def create_nobg(self):
        """NAME_ionospectra<n>_nobg.txt of each selected spectrum, in the folder of the data group;
        Remove background off: background 0, NAME_ionospectra<n>_norm.txt."""
        on = self.v_bgon.get()
        bg = self.bg_item() if on else None
        sel = sorted(self.selected(), key=lambda x: x["k"])
        if on and bg is None:
            messagebox.showerror("Background", "Choose the number of the background spectrum (1 – %d)."
                                 % len(self.items))
            return
        if not sel:
            messagebox.showerror("Background", "Select the spectra in the list (Select all for all of them).")
            return
        folder = self.v_dir.get().strip()
        flu, why = self.flu_info()
        paths = [os.path.join(folder, nobg_name(it, on)) for it in sel]
        old = [p for p in paths if os.path.exists(p)]
        if old and not messagebox.askyesno("Background", "%d of the %d files already exist (e.g. %s).\n"
                                           "Replace them?" % (len(old), len(paths), os.path.basename(old[0]))):
            return
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        q_acc, q = {}, 0.0                   # charge from the start of the experiment to the end of each
        for it in self.items:
            q += it["charge"] or 0.0
            q_acc[it["k"]] = q
        try:
            for it, p in zip(sel, paths):
                write_nobg(p, it, bg, flu, q_acc[it["k"]])
        except Exception as e:
            messagebox.showerror("Background", "Error writing %s:\n%s" % (p, e))
            return
        finally:
            self.root.configure(cursor="")
        noq = sum(1 for it in sel if not it["charge"])
        self.l_bgmsg.configure(text="%d file(s) written (%s) in %s: %s%s%s" % (
            len(paths), "background #%d" % (bg["k"] + 1) if bg else "no background removed", folder,
            os.path.basename(paths[0]) + (" …" if len(paths) > 1 else ""),
            "\n%d without current data: no normalized columns" % noq if noq else "",
            "\nno fluence column (%s)" % why if not flu else ""), foreground="#070" if flu else "#b60")
        if on and os.path.normcase(self.t_dir.get().strip()) == os.path.normcase(folder) \
                and self.t_name.get().strip() == self.v_name.get().strip():
            self.t_load(quiet=True)          # Data treatment shows this group: list again

    def show_info(self, ncurr):
        it = self.items
        n = len(it)
        self.inf["nspec"].configure(text=str(n))
        self.inf["ncurr"].configure(text=str(ncurr) + ("" if ncurr == n else "   (≠ number of spectra)"),
                                    foreground="#333" if ncurr == n else "#c00")
        if not it:
            for k in ("tspec", "ttot", "xyz", "date"):
                self.inf[k].configure(text="—")
            return
        dts = sorted({x["dt"] for x in it})
        dt = it[0]["dt"]
        self.inf["tspec"].configure(text="%g s" % dt + ("" if len(dts) == 1 else "   (varies: %g – %g s)"
                                                         % (dts[0], dts[-1])))
        tot = n * dt        # number of spectra × time of the 1st line of the (first) spectrum
        txt = "%d × %g s = %g s  (%s)" % (n, dt, tot, hms(tot))
        if len(dts) > 1:
            txt += "\nsum of the spectra: %g s" % sum(x["dt"] for x in it)
        self.inf["ttot"].configure(text=txt)
        xyz = {x["header"]["xyz"] for x in it}
        p = it[0]["header"]["xyz"]
        self.inf["xyz"].configure(text=("%g  %g  %g" % p if p else "—") + ("" if len(xyz) == 1 else "   (varies)"))
        dates = sorted({x["header"]["date"] for x in it})
        self.inf["date"].configure(text=", ".join(dates) or "—")

    # ------------------------------------------------ plots
    def selected(self):
        return [self.items[int(i)] for i in self.tv.selection()] if self.items else []

    def draw(self):
        self.fig.clear()
        sel = sorted(self.selected(), key=lambda x: x["k"])
        show_cur = self.v_cur.get()
        if show_cur:
            gs = self.fig.add_gridspec(2, 1, height_ratios=[2, 1])     # current plot = 1/3 of the height
            ax = self.fig.add_subplot(gs[0])
            axc = self.fig.add_subplot(gs[1])
        else:
            ax = self.fig.add_subplot(111)
            axc = None
        n = len(sel)
        if n <= len(TAB10):
            colors = [TAB10[i] for i in range(n)]
        else:
            ks = [x["k"] for x in sel]
            lo, hi = min(ks), max(ks)
            colors = [cm.viridis((x["k"] - lo) / max(hi - lo, 1)) for x in sel]
        mode = NORM_MODES.index(self.v_normk.get()) if self.v_normk.get() in NORM_MODES else 0
        flu, why = self.flu_info()
        bg = self.bg_item() if self.v_bgon.get() else None
        skipped = 0
        notes = []
        if mode == 2 and not flu:
            notes.append("fluence: %s" % why)
            sel_draw = []
        else:
            sel_draw = sel
        for it, c in zip(sel_draw, colors):
            y = it["y"]
            if bg is not None:
                y = y - background_of(it, bg)
            if mode:
                if not it["charge"]:
                    skipped += 1        # no current file / readings: cannot be normalized
                    continue
                y = y / (it["charge"] * 1e9) if mode == 1 else y / (it["charge"] * flu[0] / FLUENCE_UNIT)
            ax.plot(it["x"], y, color=c, lw=1, label="%d  (%g s)" % (it["k"] + 1, it["t0"]))
        ax.set_xlabel("Wavelength (nm)")
        inten = "Intensity" + (" − background" if bg is not None else "")
        ax.set_ylabel(("(%s) / charge (counts/nC)" if mode == 1 else
                       "(%s) / fluence (counts per 10¹² ions/cm²)" if mode == 2 else "%s (counts)") % inten
                      if bg is not None or not mode else
                      ("Intensity / charge (counts/nC)" if mode == 1 else
                       "Intensity / fluence (counts per 10¹² ions/cm²)"))
        if skipped:
            notes.append("%d spectra without current data not drawn" % skipped)
        if self.v_bgon.get() and self.items and bg is None:
            notes.append("background: no spectrum #%s (not removed)" % self.v_bg.get())
        elif bg is not None:
            ax.axhline(0, color="#888", lw=0.7)
        if notes:
            ax.text(0.01, 0.98, "\n".join(notes), color="#c00", transform=ax.transAxes, va="top", fontsize=8)
        if self.v_log.get():
            ax.set_yscale("log", nonpositive="mask")
        if self.v_art.get():                # artefacts: lines of the air ionized by the beam
            tr = ax.get_xaxis_transform()
            for wl, rel, band in AIR_LINES:
                if rel < AIR_LINES_MIN:     # only the strongest: 315.9, 337.1, 357.7, 380.5, 391.4
                    continue
                c = "#7b2cbf" if band.startswith("2P") else "#e8590c"
                ax.axvline(wl, color=c, ls=":", lw=0.8 + 1.0 * rel / 100, alpha=0.6 + 0.4 * rel / 100,
                           label="_nolegend_", zorder=0)
                ax.text(wl, 0.995, "%.1f" % wl, transform=tr, rotation=90, ha="right", va="top", fontsize=7,
                        color=c)
            ax.text(0.99, 0.02, "air artefacts: N₂ 2P (violet), N₂⁺ 1N (orange)", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color="#555")
        if not self.items:
            ax.text(0.5, 0.5, "Select a file of a data group (left)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
        elif not sel:
            ax.text(0.5, 0.5, "Select spectra in the list", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
        elif n <= 12 and sel_draw:
            ax.legend(fontsize=8, title="spectrum (start)", title_fontsize=8)
        else:
            ax.set_title("%d spectra (colour = time)" % n, fontsize=9)
        ax.grid(alpha=0.3)

        if axc is not None:
            low = LOWER_PLOTS.index(self.v_low.get()) if self.v_low.get() in LOWER_PLOTS else 0
            if low == 2 and not flu:
                axc.text(0.5, 0.5, "Fluence: " + why, ha="center", va="center", transform=axc.transAxes,
                         color="#c00")
                traces = {}
            elif low:
                traces = self.accumulated_traces()
                scale = 1e9 if low == 1 else flu[0]        # C -> nC or ions/cm²
                traces = {k: (t, q * scale) for k, (t, q) in traces.items()}
            else:
                traces = {it["k"]: self.current_trace(it) for it in self.items}
            if self.v_all.get():
                for t, i in traces.values():
                    if t is not None:
                        axc.plot(t, i, color="#bbbbbb", lw=0.7)
            for it, c in zip(sel, colors):
                t, i = traces.get(it["k"], (None, None))
                if t is not None:
                    axc.plot(t, i, color=c, lw=1.5 if low else 1)
                    if n <= 12:
                        axc.axvspan(it["t0"], it["t0"] + it["dt"], color=c, alpha=0.08, lw=0)
            axc.set_xlabel("Time (s)")
            axc.set_ylabel(("Beam current (nA)", "Charge (nC)", "Fluence (ions/cm²)")[low])
            axc.grid(alpha=0.3)
            if self.items and not any(x["current"] is not None for x in self.items):
                axc.text(0.5, 0.5, "No current files", ha="center", va="center",
                         transform=axc.transAxes, color="#888")
        self.canvas.draw_idle()

    def accumulated_traces(self):
        """{k: (time s, charge accumulated since the start of the experiment, C)} of the readings of
        each spectrum (each reading = I × integration time / number of readings)."""
        out, q0 = {}, 0.0
        for it in self.items:
            c = it["current"]
            if c is None or not len(c):
                out[it["k"]] = (None, None)
                continue
            t, _ = self.current_trace(it)
            dt = it["dt"] or len(c) * 0.01
            out[it["k"]] = (t, q0 + np.cumsum(c) * dt / len(c))
            q0 += float(np.sum(c)) * dt / len(c)
        return out

    @staticmethod
    def current_trace(it):
        """Time (s, from the start of the experiment) and current (nA) of the readings of one
        spectrum, spread evenly over its integration time."""
        c = it["current"]
        if c is None or not len(c):
            return None, None
        dt = it["dt"] or len(c) * 0.01
        t = it["t0"] + (np.arange(len(c)) + 0.5) * dt / len(c)
        return t, c * 1e9


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
