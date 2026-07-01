#!/usr/bin/env python3
"""
Signal efficiency for Z -> l l V, V -> h+ h-, from the private-signal NanoAODs.

The private_signals/ files are UNSKIMMED (every event is one generated Z->llV signal;
only 18-40% contain a reconstructed candidate), so the efficiency denominator is simply the
number of generated events (Events-tree entries) and

    efficiency = (events with >=1 selected candidate) / (generated events).

An event-level cutflow is reported per process:
    generated -> >=1 candidate -> + V-mass window -> + Z-mass window .
Efficiencies carry binomial (Wilson-free, sqrt(eps(1-eps)/N)) uncertainties.

Run (needs cmsenv):
    python3 signal_efficiency.py                 # all four, text table + eff_signal.png
    python3 signal_efficiency.py -p ZToPhimumu
"""
import ROOT, glob, math, argparse

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT()

SIG_DIR = "/uscms_data/d3/sitianq/zllv/private_signals"
OUT_DIR = "/uscms_data/d3/sitianq/zllv/plots"

# process -> candidate collection, V species
PROCS = {
    "ZToPhimumu": ("ZToMuMuPhi", "phi"),
    "ZToRhomumu": ("ZToMuMuRho", "rho"),
    "ZToPhiee":   ("ZToEEPhi",   "phi"),
    "ZToRhoee":   ("ZToEERho",   "rho"),
}
# V-mass window [GeV] used to define a "signal-like" ditrack candidate
V_WIN = {"phi": (1.005, 1.035), "rho": (0.60, 0.95)}
Z_WIN = (80.0, 100.0)   # fitted_mass (llV) window

def binom_err(k, n):
    if n == 0: return 0.0
    e = k / n
    return math.sqrt(max(e * (1 - e), 0.0) / n)

def cutflow(proc):
    coll, V = PROCS[proc]
    vlo, vhi = V_WIN[V]
    fs = sorted(glob.glob(f"{SIG_DIR}/{proc}_*.root"))
    d = ROOT.RDataFrame("Events", ROOT.std.vector("string")(fs))

    # event-level "any candidate passing ..." flags
    d = d.Define("any_cand",  f"n{coll} > 0")
    d = d.Define("in_vwin",   f"Sum({coll}_m_ditrack > {vlo} && {coll}_m_ditrack < {vhi}) > 0")
    d = d.Define("in_vz",     f"Sum({coll}_m_ditrack > {vlo} && {coll}_m_ditrack < {vhi} && "
                              f"{coll}_fitted_mass > {Z_WIN[0]} && {coll}_fitted_mass < {Z_WIN[1]}) > 0")

    n_gen  = d.Count()
    n_cand = d.Filter("any_cand").Count()
    n_vwin = d.Filter("in_vwin").Count()
    n_vz   = d.Filter("in_vz").Count()
    return [("generated",        int(n_gen.GetValue())),
            (">=1 candidate",    int(n_cand.GetValue())),
            ("+ V-mass window",  int(n_vwin.GetValue())),
            ("+ Z-mass window",  int(n_vz.GetValue()))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--procs", nargs="+", default=list(PROCS), choices=list(PROCS))
    args = ap.parse_args()

    import os; os.makedirs(OUT_DIR, exist_ok=True)
    finals = {}
    for proc in args.procs:
        coll, V = PROCS[proc]
        flow = cutflow(proc)
        ngen = flow[0][1]
        print(f"\n=== {proc}  ({coll}, {V}, Vwin={V_WIN[V]}, Zwin={Z_WIN}) ===")
        print(f"    {'stage':18s} {'events':>9s} {'abs.eff':>12s} {'rel.eff':>9s}")
        prev = ngen
        for stage, n in flow:
            abse = n / ngen if ngen else 0
            rele = n / prev if prev else 0
            err = binom_err(n, ngen)
            print(f"    {stage:18s} {n:9d} {abse:8.4f} +/- {err:.4f} {rele:8.3f}")
            prev = n
        finals[proc] = (flow[-1][1] / ngen, binom_err(flow[-1][1], ngen))

    # summary + bar chart of the final (full-selection) efficiency
    print("\n=== FINAL signal efficiency (selected / generated) ===")
    h = ROOT.TH1D("eff_signal", ";;signal efficiency", len(finals), 0, len(finals))
    for i, (proc, (e, de)) in enumerate(finals.items(), 1):
        print(f"    {proc:12s}  {e:.4f} +/- {de:.4f}")
        h.SetBinContent(i, e); h.SetBinError(i, de); h.GetXaxis().SetBinLabel(i, proc)
    c = ROOT.TCanvas("c", "c", 700, 500)
    h.SetBarWidth(0.6); h.SetBarOffset(0.2); h.SetFillColor(ROOT.kAzure+1)
    h.SetMarkerSize(0); h.SetMaximum(max(e for e, _ in finals.values()) * 1.3)
    h.Draw("bar e")
    c.SaveAs(f"{OUT_DIR}/eff_signal.png")
    fo = ROOT.TFile(f"{OUT_DIR}/eff_signal.root", "RECREATE"); h.Write(); fo.Close()
    print(f"\nWrote {OUT_DIR}/eff_signal.png (+ .root)")

if __name__ == "__main__":
    main()
