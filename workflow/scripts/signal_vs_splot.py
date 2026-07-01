#!/usr/bin/env python3
"""
Kinematic comparison for the phi channel (mu-mu prioritised):
    private SIGNAL MC (real-phi candidates)   vs   sPlot KAON-weighted DATA.

The data side runs the phi ditrack-mass sPlot (Voigtian phi + Bernstein bkg) and weights every
candidate by its signal sWeight -- i.e. the "real phi" component of the data. The signal side
takes real-phi candidates (ditrack in the phi window) from the private ZToPhimumu sample. Each
kinematic variable is overlaid, unit-normalised, so a shape mismatch says the data's real phi's
are NOT kinematically the Z->mumu phi signal (they'd be combinatorial phi + independent dilepton).

genWeight: the private NanoAODs do NOT store a genWeight branch (dropped by the slim producer;
only LHE_* structural branches remain), so signal is unit-weighted -- fine for normalised shapes
of a single LO process. Pass --gen-weight <name> if a weight branch is added later.

Run (needs cmsenv):
    python3 signal_vs_splot.py                       # ZToMuMuPhi
    python3 signal_vs_splot.py --mll-max 80          # add the dilepton cut to both sides
"""
import ROOT, glob, argparse, numpy as np
import data_splot as ds   # reuse CFG + build_model + Z range (same dir)

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

SIG_DIR = "/uscms_data/d3/sitianq/zllv/private_signals"
OUT_DIR = "/uscms_data/d3/sitianq/zllv/plots"
SIG_GLOB = {"ZToMuMuPhi": "ZToPhimumu_*.root", "ZToEEPhi": "ZToPhiee_*.root"}
PHI_WIN = (1.005, 1.035)   # real-phi ditrack window for the signal side

# branch suffix, nbins, lo, hi, x-title
VARS = [
    ("fitted_pt",   40, 0,   120, "p_{T}(#mu#muV) [GeV]"),
    ("fitted_eta",  40, -3,  3,   "#eta(#mu#muV)"),
    ("dilep_pt",    40, 0,   120, "p_{T}(#mu#mu) [GeV]"),
    ("ditrack_pt",  40, 0,   40,  "p_{T}(V=KK) [GeV]"),
    ("mll_fullfit", 40, 0,   120, "m(#mu#mu) [GeV]"),
    ("cos2D",       40, -1,  1,   "cos#theta_{2D}"),
    ("l_xy_sig",    40, 0,   10,  "L_{xy}/#sigma"),
    ("svprob",      40, 0,   1,   "SV fit prob."),
]
VNAMES = [v[0] for v in VARS]

def read(files, coll, extra, cap, keep_mask_fn):
    """Read m_ditrack, fitted_mass, the VARS (+ any `extra`) per file, flatten, mask, cap.
    Returns dict {branch_suffix: aligned float64 array}. `keep_mask_fn(cols)` builds the row mask."""
    want = ["m_ditrack", "fitted_mass"] + VNAMES + extra
    acc = {w: [] for w in want}
    n = 0
    for f in files:
        d = ROOT.RDataFrame("Events", f).Filter(f"n{coll}>0")
        cols = d.AsNumpy([f"{coll}_{w}" for w in want])
        if len(cols[f"{coll}_m_ditrack"]) == 0:
            continue
        flat = {w: np.concatenate([np.asarray(v, "f8") for v in cols[f"{coll}_{w}"]]) for w in want}
        del cols
        sel = keep_mask_fn(flat)
        for w in want:
            acc[w].append(flat[w][sel])
        n += int(sel.sum())
        if cap and n >= cap:
            break
    out = {w: np.concatenate(a) if a else np.array([]) for w, a in acc.items()}
    if cap and out["m_ditrack"].size > cap:
        out = {w: v[:cap] for w, v in out.items()}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--collection", default="ZToMuMuPhi", choices=list(SIG_GLOB))
    ap.add_argument("--data-files", type=int, default=12)
    ap.add_argument("--sig-files", type=int, default=20)
    ap.add_argument("--cap", type=int, default=180000)
    ap.add_argument("--mll-max", type=float, default=999.0, help="optional m(ll) cut on BOTH sides")
    ap.add_argument("--gen-weight", default="", help="signal per-event weight branch (absent by default)")
    args = ap.parse_args()
    import os; os.makedirs(OUT_DIR, exist_ok=True)
    coll = args.collection
    cfg = ds.CFG[coll]
    mllmax = args.mll_max
    tag = "" if mllmax >= 200 else f"_mll{int(mllmax)}"

    # ---- DATA: read (with alignment), fit phi, sPlot -> kaon sWeights ----
    def data_mask(c):
        return ((c["m_ditrack"] > cfg["mlo"]) & (c["m_ditrack"] < cfg["mhi"])
                & (c["fitted_mass"] > ds.Z_LO) & (c["fitted_mass"] < ds.Z_HI)
                & (c["mll_fullfit"] < mllmax))
    dfiles = sorted(glob.glob(f"{ds.EOS_BASE}/{cfg['pd']}/crab_*/*/0000/zllv_nano_*.root"))[:args.data_files]
    D = read(dfiles, coll, [], args.cap, data_mask)
    nd = D["m_ditrack"].size
    print(f"[{coll}{tag}] data candidates = {nd}")

    mtt = ROOT.RooRealVar("m_ditrack", "", cfg["mlo"], cfg["mhi"])
    mzv = ROOT.RooRealVar("fitted_mass", "", ds.Z_LO, ds.Z_HI)
    dset = ROOT.RooDataSet.from_numpy(
        {"m_ditrack": np.ascontiguousarray(D["m_ditrack"]),
         "fitted_mass": np.ascontiguousarray(D["fitted_mass"])}, [mtt, mzv])
    model, sig, bkg, nsig, nbkg, shape, width = ds.build_model(mtt, cfg, nd)
    model.fitTo(dset, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1))
    print(f"  phi fit: Nsig={nsig.getVal():.0f}+/-{nsig.getError():.0f}")
    for p in shape:
        p.setConstant(True)
    sData = ROOT.RooStats.SPlot("sData", "sData", dset, model, ROOT.RooArgList(nsig, nbkg))
    wsig = np.array([sData.GetSWeight(i, "nsig") for i in range(dset.numEntries())])

    # ---- SIGNAL: real-phi candidates, unit (or gen) weight ----
    extra = [args.gen_weight] if args.gen_weight else []
    def sig_mask(c):
        return ((c["m_ditrack"] > PHI_WIN[0]) & (c["m_ditrack"] < PHI_WIN[1])
                & (c["fitted_mass"] > ds.Z_LO) & (c["fitted_mass"] < ds.Z_HI)
                & (c["mll_fullfit"] < mllmax))
    S = read(sorted(glob.glob(f"{SIG_DIR}/{SIG_GLOB[coll]}"))[:args.sig_files], coll, extra, args.cap, sig_mask)
    ns = S["m_ditrack"].size
    wsg = S[args.gen_weight] if args.gen_weight else np.ones(ns)
    print(f"  signal real-phi candidates = {ns}"
          + ("" if args.gen_weight else "  (unit weight: no genWeight branch)"))

    # ---- overlay each variable, unit-normalised ----
    c = ROOT.TCanvas("c", "", 1400, 750); c.Divide(4, 2)
    keep = []
    for i, (v, nb, lo, hi, xt) in enumerate(VARS, 1):
        c.cd(i)
        hS = ROOT.TH1D(f"s_{v}", "", nb, lo, hi)
        hD = ROOT.TH1D(f"d_{v}", "", nb, lo, hi)
        for x, w in zip(S[v], wsg):
            hS.Fill(x, w)
        for x, w in zip(D[v], wsig):
            hD.Fill(x, w)
        for h, col in ((hS, ROOT.kAzure+1), (hD, ROOT.kBlack)):
            if h.Integral() > 0: h.Scale(1.0/h.Integral())
            h.SetLineColor(col); h.SetLineWidth(2); h.SetMarkerColor(col)
        hD.SetMarkerStyle(20); hD.SetMarkerSize(0.5)
        ymax = 1.3*max(hS.GetMaximum(), hD.GetMaximum(), 1e-9)
        hS.SetMaximum(ymax); hS.SetMinimum(0)
        hS.GetXaxis().SetTitle(xt); hS.GetYaxis().SetTitle("unit area")
        hS.SetFillColorAlpha(ROOT.kAzure+1, 0.25)
        hS.Draw("hist"); hD.Draw("e same")
        keep += [hS, hD]
    c.cd(1)
    leg = ROOT.TLegend(0.45, 0.7, 0.9, 0.9)
    leg.AddEntry(keep[0], "signal MC (real #phi)", "f")
    leg.AddEntry(keep[1], "data (#phi sWeighted)", "lep")
    leg.Draw(); keep.append(leg)
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.05)
    hdr = f"{coll}: signal vs #phi-sWeighted data" + (f", m(ll)<{int(mllmax)}" if tag else "")
    c.cd(0); t.DrawLatex(0.28, 0.965, hdr); keep.append(t)
    out = f"{OUT_DIR}/{coll}{tag}_signal_vs_splot.png"
    c.SaveAs(out)
    print(f"  wrote {out}")

if __name__ == "__main__":
    main()
