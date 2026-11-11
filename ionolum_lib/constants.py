"""Constants of ionolum (colours, atmospheres, physical constants, air lines, menus)."""

TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
# external beam: atmosphere between the exit nozzle and the sample, {element: atoms} and molar mass
# (g/mol) for the density of the gas (ideal gas)
ATMOSPHERES = {"Air": ({"N": 2 * 0.78084, "O": 2 * 0.20946, "Ar": 0.00934}, 28.965),
               "Nitrogen (N₂)": ({"N": 1.0}, 28.0134),
               "Argon": ({"Ar": 1.0}, 39.948)}
R_GAS = 8.314462618
E_CHARGE = 1.602176634e-19
FLUENCE_UNIT = 1e12          # spectra normalized to the fluence: counts per 10¹² ions/cm²
# artefacts: emission of the air ionized by the beam (radioluminescence of air, e.g. "Visualization
# of radioactive sources without gamma-radiation with UV imaging systems"): N2 second positive system
# 2P (C³Πu → B³Πg) and N2+ first negative system 1N (B²Σu+ → X²Σg+). Wavelength (nm), relative
# intensity (337.1 nm = 100, approximate: dry air at ~1 atm, AIRFLY-type values), band (v'–v'').
AIR_LINES = [
    (296.2, 5.2, "2P 3-0"), (297.7, 2.8, "2P 2-0"), (311.7, 7.2, "2P 3-2"), (313.6, 11.1, "2P 2-1"),
    (315.9, 39.3, "2P 1-0"), (328.5, 3.8, "2P 3-3"), (330.9, 2.2, "2P 2-2"), (333.9, 4.0, "2P 1-1"),
    (337.1, 100.0, "2P 0-0"), (346.9, 1.7, "2P 3-4"), (350.0, 2.8, "2P 2-3"), (353.7, 21.4, "2P 1-2"),
    (357.7, 67.4, "2P 0-1"), (367.2, 3.0, "2P 3-5"), (371.1, 5.0, "2P 2-4"), (375.5, 17.9, "2P 1-3"),
    (380.5, 27.2, "2P 0-2"), (389.4, 3.9, "2P 3-6"), (391.4, 28.0, "1N 0-0"), (394.3, 3.4, "2P 2-5"),
    (399.8, 8.4, "2P 1-4"), (405.9, 8.0, "2P 0-3"), (427.8, 5.0, "1N 0-1")]
AIR_LINES_MIN = 25.0         # lines drawn: relative intensity ≥ this
NORM_MODES = ("None", "Beam charge (counts/nC)", "Fluence (counts per 10¹² ions/cm²)")
LOWER_PLOTS = ("Beam current (nA)", "Accumulated charge (nC)", "Fluence (ions/cm²)")
SRIM_DIR = r"C:\Program Files (x86)\SRIM"
SPOT_FRACTION = 0.95        # beam diameter at the sample: circle with this fraction of the ions
