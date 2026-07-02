#!/usr/bin/env python3
"""
Template fit of m(ll phi) in the hard-cut region -> extract the Z -> mu mu phi yield.

After the discriminating cuts (real-phi window, pT(phi)>PT, Iso(phi)<ISO, m(mu mu)<MLL) the signal
is a peak at m(ll phi) ~ 91 (shape fixed from signal MC, a Gaussian) sitting on a smooth combinatorial
background (Bernstein). We fit the DATA m(ll phi) with nsig*signal + nbkg*bkg and read off nsig and
its significance. No signal-cross-section assumption -> nsig is the measured yield (a limit if ~0).

    python3 mllphi_fit.py                       # default cuts (pt>10, iso<5.5, mll<80)
    python3 mllphi_fit.py --mll-max 70 --iso-max 4
"""
import ROOT, glob, argparse, numpy as np

ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
SIG_DIR = "/uscms_data/d3/sitianq/zllv/private_signals"
CACHE   = "/uscms_data/d3/sitianq/zllv/skim_cache/Muon0_ZToMuMuPhi.npz"
OUT_DIR = "/uscms_data/d3/sitianq/zllv/plots"
COLL, PROC = "ZToMuMuPhi", "ZToPhimumu"
PHI = (1.008, 1.032)
M_LO, M_HI, NB = 70.0, 110.0, 40

def load_sig(files, pt, iso, mll):
    br = [f"{COLL}_{v}" for v in ("m_ditrack","fitted_mass","mll_fullfit","ditrack_pt","v_iso")]
    out = []
    for fn in files:
        d = ROOT.RDataFrame("Events", fn).Filter(f"n{COLL}>0")
        c = d.AsNumpy(br)
        if len(c[br[0]]) == 0: continue
        a = {v: np.concatenate([np.asarray(x,"f8") for x in c[f"{COLL}_{v}"]])
             for v in ("m_ditrack","fitted_mass","mll_fullfit","ditrack_pt","v_iso")}
        sel = ((a["m_ditrack"]>PHI[0])&(a["m_ditrack"]<PHI[1])&(a["ditrack_pt"]>pt)
               &(a["v_iso"]<iso)&(a["mll_fullfit"]<mll)&(a["fitted_mass"]>M_LO)&(a["fitted_mass"]<M_HI))
        out.append(a["fitted_mass"][sel])
    return np.concatenate(out)

def load_dat(pt, iso, mll):
    d = np.load(CACHE)
    sel = ((d["m_ditrack"]>PHI[0])&(d["m_ditrack"]<PHI[1])&(d["ditrack_pt"]>pt)
           &(d["v_iso"]<iso)&(d["mll_fullfit"]<mll)&(d["fitted_mass"]>M_LO)&(d["fitted_mass"]<M_HI))
    return d["fitted_mass"][sel]

def th1(name, arr):
    h = ROOT.TH1D(name, "", NB, M_LO, M_HI)
    for x in arr: h.Fill(x)
    return h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-min", type=float, default=10.0)
    ap.add_argument("--iso-max", type=float, default=5.5)
    ap.add_argument("--mll-max", type=float, default=80.0)
    ap.add_argument("--sig-files", type=int, default=40)
    ap.add_argument("--bkg-order", type=int, default=3)
    args = ap.parse_args()
    import os; os.makedirs(OUT_DIR, exist_ok=True)

    s = load_sig(sorted(glob.glob(f"{SIG_DIR}/{PROC}_*.root"))[:args.sig_files],
                 args.pt_min, args.iso_max, args.mll_max)
    dat = load_dat(args.pt_min, args.iso_max, args.mll_max)
    print(f"cuts: pT(phi)>{args.pt_min}, iso<{args.iso_max}, m(ll)<{args.mll_max}")
    print(f"signal MC in region: {s.size}   data in region: {dat.size}")

    mz = ROOT.RooRealVar("mz", "m(ll#phi) [GeV]", M_LO, M_HI)
    # --- signal shape: Gaussian fixed from signal MC ---
    hs = th1("hs", s); hs_ds = ROOT.RooDataHist("hs_ds", "", ROOT.RooArgList(mz), hs)
    mean = ROOT.RooRealVar("mean", "", 91, 88, 95); sigma = ROOT.RooRealVar("sigma", "", 3, 1, 8)
    gauss = ROOT.RooGaussian("gauss", "", mz, mean, sigma)
    gauss.fitTo(hs_ds, ROOT.RooFit.PrintLevel(-1))
    print(f"  signal template: mean={mean.getVal():.2f}  sigma={sigma.getVal():.2f}")
    mean.setConstant(True); sigma.setConstant(True)

    # --- background: Bernstein(3), floating; fit data = nsig*gauss + nbkg*bkg ---
    b = [ROOT.RooRealVar(f"b{i}", f"b{i}", 0.5, 0.0, 10.0) for i in range(args.bkg_order+1)]
    bkg = ROOT.RooBernstein("bkg", "", mz, ROOT.RooArgList(*b))
    nsig = ROOT.RooRealVar("nsig", "nsig", 0.01*dat.size, -0.2*dat.size, 0.5*dat.size)
    nbkg = ROOT.RooRealVar("nbkg", "nbkg", 0.9*dat.size, 0, 1.2*dat.size)
    model = ROOT.RooAddPdf("model", "", ROOT.RooArgList(gauss, bkg), ROOT.RooArgList(nsig, nbkg))
    hd = th1("hd", dat); hd_ds = ROOT.RooDataHist("hd_ds", "", ROOT.RooArgList(mz), hd)
    res = model.fitTo(hd_ds, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1))
    ns, nse = nsig.getVal(), nsig.getError()
    print(f"  FIT: Nsig = {ns:.0f} +/- {nse:.0f}   (Nbkg = {nbkg.getVal():.0f})   status={res.status()}")
    print(f"  naive significance Nsig/err = {ns/nse:+.2f} sigma")

    fr = mz.frame(ROOT.RooFit.Title(f"{COLL}: m(ll#phi) fit  (pT>{args.pt_min:g}, iso<{args.iso_max:g}, mll<{args.mll_max:g})"))
    hd_ds.plotOn(fr, ROOT.RooFit.Name("data"))
    model.plotOn(fr, ROOT.RooFit.LineColor(ROOT.kBlue))
    model.plotOn(fr, ROOT.RooFit.Components("bkg"), ROOT.RooFit.LineStyle(2), ROOT.RooFit.LineColor(ROOT.kRed))
    model.plotOn(fr, ROOT.RooFit.Components("gauss"), ROOT.RooFit.LineColor(ROOT.kGreen+2))
    c = ROOT.TCanvas("c", "", 760, 620); fr.Draw()
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.036)
    t.DrawLatex(0.52, 0.84, f"N_{{sig}} = {ns:.0f} #pm {nse:.0f}")
    t.DrawLatex(0.52, 0.78, f"signal @ {mean.getVal():.1f}, #sigma={sigma.getVal():.1f}")
    t.DrawLatex(0.52, 0.72, f"N_{{sig}}/#sigma = {ns/nse:+.2f}")
    out = f"{OUT_DIR}/{COLL}_mllphi_fit.png"; c.SaveAs(out); print(f"  wrote {out}")

if __name__ == "__main__":
    main()
