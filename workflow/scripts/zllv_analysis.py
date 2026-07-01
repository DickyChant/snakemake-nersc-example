#!/usr/bin/env python3
"""
Offline analysis for the Z -> l l V, V -> h+ h- first-observation search.

Reads the slim BPHNano NanoAODs (per-candidate flat tables ZTo{MuMu,EE}{Phi,Rho})
and fills per-candidate histograms, overlaying signal MC (private_signals/) against
collision data (the zllv_2026Jun19 CRAB output on EOS). Writes PNG overlays + a ROOT
file of all histograms.

Run (needs cmsenv first):
    export SCRAM_ARCH=el9_amd64_gcc12
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    cd /uscms_data/d3/sitianq/bph_nano/CMSSW_15_0_15/src && eval $(scramv1 runtime -sh)
    cd /uscms_data/d3/sitianq/zllv
    python3 zllv_analysis.py                       # all four collections, capped data
    python3 zllv_analysis.py -c ZToMuMuPhi         # one collection
    python3 zllv_analysis.py --max-data-files 200  # use more data
    python3 zllv_analysis.py --no-data             # signal MC only (fast)

Histograms are per-CANDIDATE (every entry of the jagged branch), not per-event; the
sample is already single-lepton-triggered and skimmed (>=1 candidate) upstream.
"""
import ROOT, os, glob, argparse

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT()

SIG_DIR  = "/uscms_data/d3/sitianq/zllv/private_signals"
EOS_BASE = "/eos/uscms/store/user/sitianq/zllv/ZLLV_zllv_2026Jun19"
OUT_DIR  = "/uscms_data/d3/sitianq/zllv/plots"

# collection -> signal glob, data primary dataset, V species
COLLECTIONS = {
    "ZToMuMuPhi": dict(sig="ZToPhimumu_*.root", pd="Muon0",   V="phi"),
    "ZToMuMuRho": dict(sig="ZToRhomumu_*.root", pd="Muon0",   V="rho"),
    "ZToEEPhi":   dict(sig="ZToPhiee_*.root",   pd="EGamma0", V="phi"),
    "ZToEERho":   dict(sig="ZToRhoee_*.root",   pd="EGamma0", V="rho"),
}
V_RANGE = {  # ditrack-mass window per V species
    "phi": (0.98, 1.10, "m(KK) [GeV]"),
    "rho": (0.30, 1.30, "m(#pi#pi) [GeV]"),
}

def observables(V):
    vlo, vhi, vlab = V_RANGE[V]
    # branch suffix, nbins, lo, hi, x-title
    return [
        ("m_ditrack",   60, vlo, vhi, vlab),
        ("mll_fullfit", 60, 0.0, 120., "m(ll) [GeV]"),
        ("fitted_mass", 60, 60., 120., "m(llV) [GeV]  (Z candidate)"),
        ("fitted_pt",   50, 0.0, 100., "p_{T}(llV) [GeV]"),
        ("svprob",      50, 0.0, 1.0,  "SV fit prob."),
        ("l_xy_sig",    50, 0.0, 10.,  "L_{xy}/#sigma"),
        ("cos2D",       50, -1., 1.0,  "cos#theta_{2D}"),
    ]

def data_files(pd, cap):
    fs = sorted(glob.glob(f"{EOS_BASE}/{pd}/crab_*/*/0000/zllv_nano_*.root"))
    n = len(fs)
    if cap and len(fs) > cap:
        fs = fs[:cap]
    return fs, n

def make_rdf(files, coll):
    """RDataFrame over `files`, keeping only events with >=1 candidate of `coll`."""
    if not files:
        return None, 0
    df = ROOT.RDataFrame("Events", ROOT.std.vector("string")(files))
    df = df.Filter(f"n{coll} > 0")
    return df, df.Count()

def fill(df, coll, obs):
    """Book one Histo1D per observable; returns {name: (lazy TH1D, model)}."""
    hists = {}
    for suf, nb, lo, hi, xlab in obs:
        model = ROOT.RDF.TH1DModel(f"{coll}_{suf}", f";{xlab};candidates", nb, lo, hi)
        hists[suf] = (df.Define("flat", f"{coll}_{suf}").Histo1D(model, "flat"), xlab)
    return hists

def draw(coll, suf, xlab, h_sig, h_dat, outroot):
    c = ROOT.TCanvas("c", "c", 700, 600)
    leg = ROOT.TLegend(0.62, 0.75, 0.88, 0.88)
    drawn = []
    for h, col, lab, opt in ((h_sig, ROOT.kAzure+1, "signal MC", "hist"),
                             (h_dat, ROOT.kBlack,   "data",      "e")):
        if h is None or h.Integral() <= 0:
            continue
        h = h.Clone(f"{coll}_{suf}_{lab.split()[0]}")
        h.Scale(1.0 / h.Integral())          # shape comparison (no lumi norm yet)
        h.SetLineColor(col); h.SetMarkerColor(col); h.SetLineWidth(2)
        if opt == "e": h.SetMarkerStyle(20); h.SetMarkerSize(0.7)
        outroot.cd(); h.Write()
        drawn.append((h, opt)); leg.AddEntry(h, lab, "lep" if opt == "e" else "l")
    if not drawn:
        return
    ymax = max(h.GetMaximum() for h, _ in drawn) * 1.35
    first = True
    for h, opt in drawn:
        h.SetMaximum(ymax); h.SetMinimum(0)
        h.Draw(opt if first else opt + " same"); first = False
    leg.Draw()
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.035)
    t.DrawLatex(0.12, 0.92, f"{coll}   ({suf})")
    c.SaveAs(f"{OUT_DIR}/{coll}_{suf}.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--collections", nargs="+", default=list(COLLECTIONS),
                    choices=list(COLLECTIONS), help="which collections to run")
    ap.add_argument("--max-data-files", type=int, default=40,
                    help="cap on data files per PD (0 = all). Logged if it truncates.")
    ap.add_argument("--no-data", action="store_true", help="signal MC only")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    outroot = ROOT.TFile(f"{OUT_DIR}/zllv_hists.root", "RECREATE")
    print(f"{'collection':12s} {'sig cand':>10s} {'data cand':>10s}   data files used")
    print("-" * 60)

    for coll in args.collections:
        info = COLLECTIONS[coll]
        obs = observables(info["V"])

        sig_files = sorted(glob.glob(f"{SIG_DIR}/{info['sig']}"))
        df_sig, n_sig = make_rdf(sig_files, coll)
        h_sig = fill(df_sig, coll, obs) if df_sig is not None else {}

        h_dat, ndat_lazy, used, total = {}, None, 0, 0
        if not args.no_data:
            dfiles, total = data_files(info["pd"], args.max_data_files)
            used = len(dfiles)
            df_dat, ndat_lazy = make_rdf(dfiles, coll)
            if df_dat is not None:
                h_dat = fill(df_dat, coll, obs)

        # trigger the event loops (lazy until a value is read)
        ns = int(n_sig.GetValue()) if n_sig else 0
        nd = int(ndat_lazy.GetValue()) if ndat_lazy else 0
        trunc = f"{used}/{total}" + ("  (TRUNCATED)" if total > used else "")
        print(f"{coll:12s} {ns:10d} {nd:10d}   {trunc if not args.no_data else '-- (--no-data)'}")

        for suf, *_ in obs:
            hs = h_sig.get(suf, (None,))[0]
            hd = h_dat.get(suf, (None,))[0]
            xlab = dict((s, x) for s, _, _, _, x in obs)[suf]
            draw(coll, suf, xlab, hs, hd, outroot)

    outroot.Close()
    print(f"\nWrote PNGs + histograms to {OUT_DIR}/  (zllv_hists.root)")

if __name__ == "__main__":
    main()
