#!/usr/bin/env python3
"""
Full-stats m(ll phi) bump hunt on the skim cache (Muon0 / ZToMuMuPhi).

Loads the merged skim, applies the analysis cuts (real-phi window, m(ll)<mll_max, ditrack pt,
svprob), histograms the 3-body mass m(ll phi)=fitted_mass, fits a smooth background EXCLUDING a
signal window around the Z, and tests for a localized excess at 91 (the Z -> mu mu phi signature).
Produces a spectrum + background overlay and a background-subtracted residual, and prints the
integrated-excess significance.

    python3 cache_bump.py                    # default cuts
    python3 cache_bump.py --mll-max 70 --pt-min 8 --svprob-min 0.2
"""
import argparse, numpy as np, ROOT

ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)
CACHE = "/uscms_data/d3/sitianq/zllv/skim_cache/Muon0_ZToMuMuPhi.npz"
OUT_DIR = "/uscms_data/d3/sitianq/zllv/plots"
M_LO, M_HI, NB = 70.0, 110.0, 40
SR = (86.0, 96.0)          # signal window (excluded from the bkg fit)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-lo", type=float, default=1.008)
    ap.add_argument("--phi-hi", type=float, default=1.032)
    ap.add_argument("--mll-max", type=float, default=80.0)
    ap.add_argument("--pt-min", type=float, default=5.0)
    ap.add_argument("--svprob-min", type=float, default=0.1)
    ap.add_argument("--degree", type=int, default=5, help="polynomial degree for the bkg fit")
    args = ap.parse_args()
    import os; os.makedirs(OUT_DIR, exist_ok=True)

    d = np.load(CACHE)
    m, z, mll, pt, sv = (d["m_ditrack"], d["fitted_mass"], d["mll_fullfit"],
                         d["ditrack_pt"], d["svprob"])
    cut = ((m > args.phi_lo) & (m < args.phi_hi) & (mll < args.mll_max)
           & (pt > args.pt_min) & (sv > args.svprob_min))
    zz = z[cut]
    print(f"candidates after cuts: {zz.size}")

    h, edges = np.histogram(zz, bins=NB, range=(M_LO, M_HI))
    ctr = 0.5 * (edges[:-1] + edges[1:])
    err = np.sqrt(h)
    inSR = (ctr >= SR[0]) & (ctr <= SR[1])
    fitm = ~inSR & (h > 0)
    # weighted polynomial background fit on the sideband bins
    coef = np.polyfit(ctr[fitm], h[fitm], args.degree, w=1.0 / err[fitm])
    bkg = np.polyval(coef, ctr)

    obs, exp = h[inSR].sum(), bkg[inSR].sum()
    signif = (obs - exp) / np.sqrt(exp) if exp > 0 else 0.0
    print(f"  SR {SR}: obs={obs:.0f}  exp(bkg)={exp:.0f}  excess={obs-exp:+.0f}  "
          f"significance={signif:+.1f} sigma")

    c = ROOT.TCanvas("c", "", 760, 800)
    p1 = ROOT.TPad("p1", "", 0, 0.32, 1, 1); p1.SetBottomMargin(0.02); p1.Draw()
    p2 = ROOT.TPad("p2", "", 0, 0.0, 1, 0.32); p2.SetTopMargin(0.03); p2.SetBottomMargin(0.3); p2.Draw()
    hh = ROOT.TH1D("hh", "", NB, M_LO, M_HI); gb = ROOT.TGraph(NB)
    hr = ROOT.TH1D("hr", "", NB, M_LO, M_HI)
    for b in range(NB):
        hh.SetBinContent(b + 1, h[b]); hh.SetBinError(b + 1, err[b])
        gb.SetPoint(b, ctr[b], bkg[b])
        r = (h[b] - bkg[b]) / err[b] if err[b] > 0 else 0
        hr.SetBinContent(b + 1, r); hr.SetBinError(b + 1, 1)
    p1.cd()
    hh.SetMarkerStyle(20); hh.SetMarkerSize(0.6); hh.SetLineColor(ROOT.kBlack)
    hh.GetYaxis().SetTitle("candidates"); hh.GetXaxis().SetLabelSize(0); hh.SetMinimum(0)
    hh.Draw("e")
    gb.SetLineColor(ROOT.kRed); gb.SetLineWidth(2); gb.Draw("l")
    for x in SR:
        ln = ROOT.TLine(x, 0, x, hh.GetMaximum()); ln.SetLineStyle(2); ln.SetLineColor(ROOT.kGreen+2); ln.Draw(); hh._k = getattr(hh, "_k", []) + [ln]
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.045)
    t.DrawLatex(0.13, 0.93, "Muon0 ZToMuMuPhi: m(ll#phi) bump hunt (full stats)")
    t.DrawLatex(0.55, 0.83, f"cuts: m(ll)<{args.mll_max:.0f}, p_{{T}}(V)>{args.pt_min:g}, svprob>{args.svprob_min:g}")
    t.DrawLatex(0.55, 0.77, f"SR excess = {obs-exp:+.0f}  ({signif:+.1f}#sigma)")
    p2.cd()
    hr.SetMarkerStyle(20); hr.SetMarkerSize(0.6); hr.SetLineColor(ROOT.kBlack)
    hr.GetYaxis().SetTitle("(data#minusbkg)/#sigma"); hr.GetYaxis().SetNdivisions(505)
    hr.GetYaxis().SetTitleSize(0.11); hr.GetYaxis().SetLabelSize(0.09); hr.GetYaxis().SetTitleOffset(0.4)
    hr.GetXaxis().SetTitle("m(ll#phi) [GeV]"); hr.GetXaxis().SetTitleSize(0.12); hr.GetXaxis().SetLabelSize(0.1)
    hr.SetMinimum(-5); hr.SetMaximum(5); hr.Draw("e")
    z0 = ROOT.TLine(M_LO, 0, M_HI, 0); z0.SetLineColor(ROOT.kRed); z0.Draw()
    out = f"{OUT_DIR}/ZToMuMuPhi_cache_bump.png"
    c.SaveAs(out)
    print(f"  wrote {out}")

if __name__ == "__main__":
    main()
