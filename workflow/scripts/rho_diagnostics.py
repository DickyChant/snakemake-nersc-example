#!/usr/bin/env python3
"""
Diagnostic: why the rho channel needs a multivariate/cut approach, not a ditrack-mass sPlot.

Compares signal MC vs data for the ZToMuMuRho (or ZToEERho) collection in three 1D
observables plus the 2D (m_ll, m_llV) plane, and prints a few separation numbers. The
headline: rho -> pi pi has no visible peak in the data ditrack mass (flat combinatorial),
so the discriminating power has to come from m(ll) being below the Z pole + track/vertex
variables, not from the V-mass shape.

Run (needs cmsenv):
    python3 rho_diagnostics.py            # ZToMuMuRho
    python3 rho_diagnostics.py -c ZToEERho
"""
import ROOT, glob, argparse, numpy as np

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

SIG_DIR  = "/uscms_data/d3/sitianq/zllv/private_signals"
EOS_BASE = "/eos/uscms/store/user/sitianq/zllv/ZLLV_zllv_2026Jun19"
OUT_DIR  = "/uscms_data/d3/sitianq/zllv/plots"

CFG = {
    "ZToMuMuRho": dict(pd="Muon0",   sig="ZToRhomumu_*.root"),
    "ZToEERho":   dict(pd="EGamma0", sig="ZToRhoee_*.root"),
}

def load(files, coll, cap):
    m, z, l = [], [], []
    for f in files:
        d = ROOT.RDataFrame("Events", f).Filter(f"n{coll}>0")
        c = d.AsNumpy([f"{coll}_m_ditrack", f"{coll}_fitted_mass", f"{coll}_mll_fullfit"])
        if len(c[f"{coll}_m_ditrack"]) == 0:
            continue
        m.append(np.concatenate([np.asarray(v, "f8") for v in c[f"{coll}_m_ditrack"]]))
        z.append(np.concatenate([np.asarray(v, "f8") for v in c[f"{coll}_fitted_mass"]]))
        l.append(np.concatenate([np.asarray(v, "f8") for v in c[f"{coll}_mll_fullfit"]]))
        if sum(len(x) for x in m) >= cap:
            break
    return np.concatenate(m), np.concatenate(z), np.concatenate(l)

def th1(name, arr, nb, lo, hi, col):
    h = ROOT.TH1D(name, "", nb, lo, hi)
    for x in arr:
        h.Fill(x)
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())
    h.SetLineColor(col); h.SetLineWidth(2); h.SetMarkerColor(col)
    return h

_KEEP = []   # module-level anchor so ROOT objects survive until SaveAs (pad proxies are transient)

def overlay(pad, hs, hd, xtitle, header):
    pad.cd()
    hd.SetTitle("")
    ymax = 1.3 * max(hs.GetMaximum(), hd.GetMaximum())
    hd.SetMaximum(ymax); hd.SetMinimum(0)
    hd.GetXaxis().SetTitle(xtitle); hd.GetYaxis().SetTitle("unit area")
    hd.Draw("hist"); hs.Draw("hist same")
    leg = ROOT.TLegend(0.62, 0.74, 0.88, 0.88)
    leg.AddEntry(hd, "data", "l"); leg.AddEntry(hs, "signal MC", "l"); leg.Draw()
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.045); t.DrawLatex(0.12, 0.93, header)
    _KEEP.extend([hd, hs, leg, t])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--collection", default="ZToMuMuRho", choices=list(CFG))
    ap.add_argument("--data-files", type=int, default=3)
    ap.add_argument("--sig-files", type=int, default=12)
    ap.add_argument("--cap", type=int, default=400000)
    args = ap.parse_args()
    import os; os.makedirs(OUT_DIR, exist_ok=True)
    cfg = CFG[args.collection]; coll = args.collection

    sm, sz, sl = load(sorted(glob.glob(f"{SIG_DIR}/{cfg['sig']}"))[:args.sig_files], coll, args.cap)
    dfiles = sorted(glob.glob(f"{EOS_BASE}/{cfg['pd']}/crab_*/*/0000/zllv_nano_*.root"))[:args.data_files]
    dm, dz, dl = load(dfiles, coll, args.cap)
    print(f"[{coll}] signal cands={sm.size}  data cands={dm.size}")
    # separation numbers
    for tag, mll in (("signal", sl), ("data", dl)):
        print(f"  {tag}: <m(ll)>={mll.mean():.1f}  frac m(ll)<80={ (mll<80).mean():.3f}")

    c = ROOT.TCanvas("c", "", 1150, 900); c.Divide(2, 2)
    overlay(c.cd(1), th1("s_v", sm, 25, 0.5, 1.0, ROOT.kAzure+1),
                     th1("d_v", dm, 25, 0.5, 1.0, ROOT.kBlack),
            "m(#pi#pi) [GeV]", "(a) ditrack mass: rho peak in MC, FLAT in data")
    overlay(c.cd(2), th1("s_l", sl, 40, 0, 120, ROOT.kAzure+1),
                     th1("d_l", dl, 40, 0, 120, ROOT.kBlack),
            "m(ll) [GeV]", "(b) dilepton mass: signal below Z, data at Z")
    overlay(c.cd(3), th1("s_z", sz, 40, 40, 140, ROOT.kAzure+1),
                     th1("d_z", dz, 40, 40, 140, ROOT.kBlack),
            "m(llV) [GeV]", "(c) m(llV): both near Z")
    # (d) 2D data: m(ll) vs m(llV) -- shows the DY band the signal must be dug out of
    p4 = c.cd(4)
    h2 = ROOT.TH2D("h2", ";m(ll) [GeV];m(llV) [GeV]", 50, 0, 120, 50, 40, 140)
    for x, y in zip(dl, dz):
        h2.Fill(x, y)
    h2.Draw("colz")
    box = ROOT.TBox(0, 85, 80, 97); box.SetFillStyle(0); box.SetLineColor(ROOT.kRed); box.SetLineWidth(2); box.Draw()
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.04)
    t.DrawLatex(0.12, 0.93, "(d) data m(ll) vs m(llV); red = signal-ish region")
    _KEEP.extend([h2, box, t])
    out = f"{OUT_DIR}/{coll}_diagnostics.png"
    c.SaveAs(out)
    print(f"  wrote {out}")

if __name__ == "__main__":
    main()
