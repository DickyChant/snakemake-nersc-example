#!/usr/bin/env python3
"""
Effect of the dilepton-mass cut on the PRIVATE SIGNAL (Z -> l l V, V -> h+ h-).

For a candidate collection, selects real-V candidates (ditrack mass in the V window) from the
matching private-signal sample and shows m(llV) with and without the m(ll) < cut. This is the
signal-side counterpart of the data sPlot: it quantifies how much true signal the cut keeps and
confirms the Z -> l l V resonance in m(llV) SURVIVES the cut (stays peaked near m_Z), unlike the
data sPlot-signal, whose m(llV) slid down with m(ll) -- i.e. the cut removes DY-like background
without removing genuine signal.

Run (needs cmsenv):
    python3 signal_mll_cut.py                       # ZToMuMuPhi, m(ll)<80
    python3 signal_mll_cut.py -c ZToEERho --mll-max 75
"""
import ROOT, glob, argparse, numpy as np

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

SIG_DIR = "/uscms_data/d3/sitianq/zllv/private_signals"
OUT_DIR = "/uscms_data/d3/sitianq/zllv/plots"

# collection -> (signal glob, V-mass window) ; windows match signal_efficiency.py
CFG = {
    "ZToMuMuPhi": ("ZToPhimumu_*.root", (1.005, 1.035), "#phi#rightarrowKK"),
    "ZToEEPhi":   ("ZToPhiee_*.root",   (1.005, 1.035), "#phi#rightarrowKK"),
    "ZToMuMuRho": ("ZToRhomumu_*.root", (0.60, 0.95),   "#rho#rightarrow#pi#pi"),
    "ZToEERho":   ("ZToRhoee_*.root",   (0.60, 0.95),   "#rho#rightarrow#pi#pi"),
}
Z_LO, Z_HI, Z_NB = 40.0, 140.0, 50

def load(files, coll, vlo, vhi, cap):
    z, l = [], []
    for f in files:
        d = ROOT.RDataFrame("Events", f).Filter(f"n{coll}>0")
        c = d.AsNumpy([f"{coll}_m_ditrack", f"{coll}_fitted_mass", f"{coll}_mll_fullfit"])
        if len(c[f"{coll}_m_ditrack"]) == 0:
            continue
        m  = np.concatenate([np.asarray(v, "f8") for v in c[f"{coll}_m_ditrack"]])
        mz = np.concatenate([np.asarray(v, "f8") for v in c[f"{coll}_fitted_mass"]])
        ml = np.concatenate([np.asarray(v, "f8") for v in c[f"{coll}_mll_fullfit"]])
        vsel = (m > vlo) & (m < vhi) & (mz > Z_LO) & (mz < Z_HI)   # real-V candidates only
        z.append(mz[vsel]); l.append(ml[vsel])
        if sum(len(x) for x in z) >= cap:
            break
    return np.concatenate(z), np.concatenate(l)

def hist(name, arr, col, style=0):
    h = ROOT.TH1D(name, "", Z_NB, Z_LO, Z_HI)
    for x in arr:
        h.Fill(x)
    h.SetLineColor(col); h.SetLineWidth(2); h.SetMarkerColor(col)
    if style:
        h.SetFillColorAlpha(col, 0.25)
    return h

def peak(h):
    b = h.GetMaximumBin()
    return h.GetBinCenter(b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--collection", default="ZToMuMuPhi", choices=list(CFG))
    ap.add_argument("--mll-max", type=float, default=80.0)
    ap.add_argument("--sig-files", type=int, default=20)
    ap.add_argument("--cap", type=int, default=400000)
    args = ap.parse_args()
    import os; os.makedirs(OUT_DIR, exist_ok=True)
    sglob, (vlo, vhi), vlab = CFG[args.collection]

    z, l = load(sorted(glob.glob(f"{SIG_DIR}/{sglob}"))[:args.sig_files],
                args.collection, vlo, vhi, args.cap)
    keep = l < args.mll_max
    eff = keep.mean() if z.size else 0.0
    hInc = hist("mllV_inc", z,        ROOT.kGray+2, style=1)
    hCut = hist("mllV_cut", z[keep],  ROOT.kAzure+1, style=1)
    print(f"[{args.collection}] real-V signal cands = {z.size}")
    print(f"  m(ll)<{args.mll_max:.0f} keeps {int(keep.sum())} ({eff:.3f}); "
          f"m(llV) peak: inclusive {peak(hInc):.0f} GeV, cut {peak(hCut):.0f} GeV")

    c = ROOT.TCanvas("c", "", 1100, 500); c.Divide(2, 1)
    # (left) absolute -- shows efficiency loss
    c.cd(1)
    hInc.SetMaximum(1.25 * hInc.GetMaximum()); hInc.SetMinimum(0)
    hInc.GetXaxis().SetTitle("m(llV) [GeV]"); hInc.GetYaxis().SetTitle("signal candidates")
    hInc.Draw("hist"); hCut.Draw("hist same")
    leg = ROOT.TLegend(0.58, 0.72, 0.88, 0.88)
    leg.AddEntry(hInc, "all real-V signal", "f")
    leg.AddEntry(hCut, f"+ m(ll)<{args.mll_max:.0f}  (#epsilon={eff:.2f})", "f")
    leg.Draw()
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.04)
    t.DrawLatex(0.12, 0.93, f"{args.collection} ({vlab}): m(llV), absolute")
    # (right) normalized -- shows the resonance shape is preserved
    c.cd(2)
    hI2, hC2 = hInc.Clone("i2"), hCut.Clone("c2")
    for h in (hI2, hC2):
        if h.Integral() > 0: h.Scale(1.0/h.Integral())
        h.SetFillStyle(0)
    hI2.SetMaximum(1.25*max(hI2.GetMaximum(), hC2.GetMaximum())); hI2.SetMinimum(0)
    hI2.GetXaxis().SetTitle("m(llV) [GeV]"); hI2.GetYaxis().SetTitle("unit area")
    hI2.Draw("hist"); hC2.Draw("hist same")
    t2 = ROOT.TLatex(); t2.SetNDC(); t2.SetTextSize(0.04)
    t2.DrawLatex(0.12, 0.93, "shape (normalized): resonance preserved")
    c._keep = (hInc, hCut, hI2, hC2, leg, t, t2)
    out = f"{OUT_DIR}/{args.collection}_signal_mllcut.png"
    c.SaveAs(out)
    print(f"  wrote {out}")

if __name__ == "__main__":
    main()
