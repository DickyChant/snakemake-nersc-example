#!/usr/bin/env python3
"""
sPlot on data for Z -> l l V, V -> h+ h-.

Procedure (RooFit + RooStats::SPlot):
  1. Load per-candidate (m_ditrack, fitted_mass) from the zllv_2026Jun19 data NanoAODs on EOS.
  2. Fit the DITRACK mass m_ditrack (the discriminating variable): signal Voigtian (phi/rho
     Breit-Wigner (x) Gaussian resolution) + smooth background (Chebyshev).
  3. Fix the shape parameters, run SPlot -> per-candidate signal sWeights.
  4. Project the sWeights onto fitted_mass (the l l V "Z" candidate) -> the background-
     subtracted Z line shape, WITHOUT ever cutting on fitted_mass.

The two sPlot variables must be uncorrelated for signal: m(hh) (V) and m(llV) (Z) are
independent by construction, so this is valid.

Run (needs cmsenv):
    python3 data_splot.py                       # ZToMuMuPhi, cap 300k candidates
    python3 data_splot.py -c ZToEEPhi
    python3 data_splot.py --max-cands 600000 --max-files 60
"""
import ROOT, glob, argparse, numpy as np

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

EOS_BASE = "/eos/uscms/store/user/sitianq/zllv/ZLLV_zllv_2026Jun19"
OUT_DIR  = "/uscms_data/d3/sitianq/zllv/plots"

# collection -> data PD, V species, ditrack fit window, V nominal mass + BW width [GeV]
CFG = {
    "ZToMuMuPhi": dict(pd="Muon0",   V="phi", mlo=1.00, mhi=1.05, m0=1.019461, gamma=0.004249),
    "ZToEEPhi":   dict(pd="EGamma0", V="phi", mlo=1.00, mhi=1.05, m0=1.019461, gamma=0.004249),
    "ZToMuMuRho": dict(pd="Muon0",   V="rho", mlo=0.50, mhi=1.00, m0=0.77526,  gamma=0.1474),
    "ZToEERho":   dict(pd="EGamma0", V="rho", mlo=0.50, mhi=1.00, m0=0.77526,  gamma=0.1474),
}
Z_LO, Z_HI, Z_NB = 40.0, 140.0, 50   # fitted_mass (llV) histogram

def load(coll, pd, mlo, mhi, max_files, max_cands, mll_lo, mll_hi, pt_min, svprob_min):
    """Read files one at a time (this node has little RAM), flattening each to plain
    float64 immediately so the heavy per-event RVec objects are released, and stop once
    max_cands candidates are collected. Candidates are kept only if m_ditrack in [mlo,mhi],
    fitted_mass in [Z_LO,Z_HI], m(ll) in [mll_lo,mll_hi], ditrack_pt > pt_min, and
    svprob > svprob_min. The preselection suppresses fakes -- Drell-Yan Z->ll at the Z pole
    (m(ll)), soft QCD/UE V (ditrack_pt), and bad-vertex combinatorics (svprob) -- while
    keeping genuine Z->llV. All three are ~uncorrelated with the projection variable m(llV),
    so the ditrack fit / sPlot stays valid; the phi shape is re-fit within the cut sample."""
    fs = sorted(glob.glob(f"{EOS_BASE}/{pd}/crab_*/*/0000/zllv_nano_*.root"))
    total = len(fs)
    cap_files = max_files if max_files else total
    br = [f"{coll}_m_ditrack", f"{coll}_fitted_mass", f"{coll}_mll_fullfit",
          f"{coll}_ditrack_pt", f"{coll}_svprob"]
    m_parts, mz_parts, n, used = [], [], 0, 0
    for f in fs[:cap_files]:
        d = ROOT.RDataFrame("Events", f).Filter(f"n{coll}>0")
        cols = d.AsNumpy(br)
        used += 1
        if len(cols[br[0]]) == 0:
            continue
        flat = [np.concatenate([np.asarray(v, dtype="float64") for v in cols[b]]) for b in br]
        m, mz, ml, pt, sv = flat
        del cols, flat
        sel = ((m > mlo) & (m < mhi) & (mz > Z_LO) & (mz < Z_HI)
               & (ml > mll_lo) & (ml < mll_hi) & (pt > pt_min) & (sv > svprob_min))
        m_parts.append(m[sel]); mz_parts.append(mz[sel]); n += int(sel.sum())
        if max_cands and n >= max_cands:
            break
    m  = np.concatenate(m_parts)
    mz = np.concatenate(mz_parts)
    if max_cands and m.size > max_cands:
        m, mz = m[:max_cands], mz[:max_cands]
    return m, mz, used, total

def build_model(mtt, cfg, n_tot):
    mean  = ROOT.RooRealVar("mean",  "mean",  cfg["m0"], cfg["m0"]-0.01, cfg["m0"]+0.01)
    width = ROOT.RooRealVar("width", "width", cfg["gamma"]); width.setConstant(True)  # PDG BW width
    sigma = ROOT.RooRealVar("sigma", "sigma", 0.002, 0.0008, 0.02)                    # detector resol.
    sig   = ROOT.RooVoigtian("sig", "sig", mtt, mean, width, sigma)
    # Bernstein background: positive-by-construction (all coeffs >=0), so the extended NLL
    # never sees a negative pdf. Order 4 handles the threshold rise + plateau.
    bcoef = [ROOT.RooRealVar(f"b{i}", f"b{i}", 0.5, 0.0, 10.0) for i in range(5)]
    bkg = ROOT.RooBernstein("bkg", "bkg", mtt, ROOT.RooArgList(*bcoef))
    nsig = ROOT.RooRealVar("nsig", "nsig", 0.1*n_tot, 0, n_tot)
    nbkg = ROOT.RooRealVar("nbkg", "nbkg", 0.9*n_tot, 0, n_tot)
    model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(sig, bkg), ROOT.RooArgList(nsig, nbkg))
    shape = [mean, sigma] + bcoef
    # keep `width` referenced too: RooVoigtian holds a bare pointer to it, so if Python GCs
    # the RooRealVar the fit dereferences freed memory and segfaults.
    return model, sig, bkg, nsig, nbkg, shape, width

def draw_fit(coll, mtt, data, model, sig, bkg, nsig, nbkg, cfg):
    fr = mtt.frame(ROOT.RooFit.Bins(120), ROOT.RooFit.Title(f"{coll}: ditrack-mass fit"))
    data.plotOn(fr, ROOT.RooFit.Name("data"))
    model.plotOn(fr, ROOT.RooFit.Name("total"), ROOT.RooFit.LineColor(ROOT.kBlue))
    model.plotOn(fr, ROOT.RooFit.Components("bkg"), ROOT.RooFit.LineStyle(2), ROOT.RooFit.LineColor(ROOT.kRed))
    model.plotOn(fr, ROOT.RooFit.Components("sig"), ROOT.RooFit.LineColor(ROOT.kGreen+2))
    c = ROOT.TCanvas("cfit", "cfit", 700, 600); fr.Draw()
    lab = ("m(KK)" if cfg["V"] == "phi" else "m(#pi#pi)") + " [GeV]"
    fr.GetXaxis().SetTitle(lab)
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.033)
    t.DrawLatex(0.55, 0.85, f"N_{{sig}} = {nsig.getVal():.0f} #pm {nsig.getError():.0f}")
    t.DrawLatex(0.55, 0.80, f"N_{{bkg}} = {nbkg.getVal():.0f} #pm {nbkg.getError():.0f}")
    c.SaveAs(f"{OUT_DIR}/{coll}_ditrack_fit.png")

def draw_splot(coll, mz_arr, sw_sig, sw_bkg):
    hR = ROOT.TH1D(f"{coll}_Zraw", ";m(llV) [GeV];candidates",       Z_NB, Z_LO, Z_HI)
    hS = ROOT.TH1D(f"{coll}_Zsig", ";m(llV) [GeV];signal sWeighted", Z_NB, Z_LO, Z_HI)
    hB = ROOT.TH1D(f"{coll}_Zbkg", ";m(llV) [GeV];bkg sWeighted",    Z_NB, Z_LO, Z_HI)
    for x, ws, wb in zip(mz_arr, sw_sig, sw_bkg):
        hR.Fill(x); hS.Fill(x, ws); hB.Fill(x, wb)
    for h, col in ((hR, ROOT.kGray+2), (hS, ROOT.kGreen+2), (hB, ROOT.kRed)):
        h.SetLineColor(col); h.SetMarkerColor(col); h.SetLineWidth(2)

    # (1) the money plot: signal-sWeighted m(llV) ALONE, on its own scale -- this is the
    #     background-subtracted Z -> l l V line shape.
    c1 = ROOT.TCanvas("cZsig", "", 720, 600)
    hS.SetMarkerStyle(20); hS.SetMarkerSize(0.7); hS.SetFillColorAlpha(ROOT.kGreen+2, 0.25)
    hS.SetMinimum(0); hS.Draw("hist e")
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.034)
    t.DrawLatex(0.13, 0.92, f"{coll}: background-subtracted m(llV)  (signal sWeighted)")
    c1.SaveAs(f"{OUT_DIR}/{coll}_splot_Zsignal.png")

    # (2) intermediate panels: raw | signal-sWeighted | bkg-sWeighted | shape overlay
    c2 = ROOT.TCanvas("csp4", "", 1100, 850); c2.Divide(2, 2)
    c2.cd(1); hR.SetFillColorAlpha(ROOT.kGray, 0.4); hR.SetMinimum(0); hR.Draw("hist")
    _title("(a) raw m(llV): all candidates")
    c2.cd(2); hS.Draw("hist e"); _title("(b) signal sWeighted -> Z peak")
    c2.cd(3); hB.SetFillColorAlpha(ROOT.kRed, 0.2); hB.SetMinimum(0); hB.Draw("hist e")
    _title("(c) background sWeighted")
    c2.cd(4)   # shape overlay (unit area) so the small signal is visible next to background
    ov = []
    leg = ROOT.TLegend(0.55, 0.7, 0.88, 0.88)
    for h, lab in ((hR, "raw"), (hS, "signal sW"), (hB, "bkg sW")):
        hh = h.Clone(h.GetName()+"_n")
        if hh.Integral() > 0: hh.Scale(1.0/hh.Integral())
        hh.SetFillStyle(0); ov.append(hh); leg.AddEntry(hh, lab, "l")
    ov[0].SetMaximum(1.3*max(h.GetMaximum() for h in ov)); ov[0].SetMinimum(0)
    ov[0].GetYaxis().SetTitle("unit area"); ov[0].Draw("hist")
    ov[1].Draw("hist same"); ov[2].Draw("hist same"); leg.Draw()
    _title("(d) shape comparison (unit area)")
    c2.SaveAs(f"{OUT_DIR}/{coll}_splot_panels.png")
    return hR, hS, hB

def _title(txt):
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.04); t.DrawLatex(0.12, 0.93, txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--collection", default="ZToMuMuPhi", choices=list(CFG))
    ap.add_argument("--max-files", type=int, default=40, help="data files (0=all)")
    ap.add_argument("--max-cands", type=int, default=300000, help="cap candidates for the fit (0=all)")
    ap.add_argument("--mll-min", type=float, default=0.0,   help="dilepton-mass lower cut [GeV]")
    ap.add_argument("--mll-max", type=float, default=999.0, help="dilepton-mass upper cut [GeV]")
    ap.add_argument("--pt-min", type=float, default=0.0,     help="ditrack (V) pT lower cut [GeV]")
    ap.add_argument("--svprob-min", type=float, default=0.0, help="SV fit-probability lower cut")
    args = ap.parse_args()
    import os; os.makedirs(OUT_DIR, exist_ok=True)
    cfg = CFG[args.collection]

    # tag outputs so a cut run does not overwrite others
    parts = []
    if args.mll_min > 0.0 or args.mll_max < 200.0: parts.append(f"mll{int(args.mll_min)}-{int(args.mll_max)}")
    if args.pt_min > 0.0:     parts.append(f"pt{args.pt_min:g}")
    if args.svprob_min > 0.0: parts.append(f"sv{args.svprob_min:g}")
    label = args.collection + ("_" + "_".join(parts) if parts else "")

    m, mz, used, total = load(args.collection, cfg["pd"], cfg["mlo"], cfg["mhi"],
                              args.max_files, args.max_cands, args.mll_min, args.mll_max,
                              args.pt_min, args.svprob_min)
    print(f"[{label}] data files {used}/{total}"
          + ("  (TRUNCATED)" if total > used else "")
          + f" ; cuts: m(ll)<{args.mll_max:.0f} pt(V)>{args.pt_min:g} svprob>{args.svprob_min:g}"
          + f" ; candidates used = {m.size}")
    if m.size < 100:
        print("too few candidates; aborting"); return

    mtt = ROOT.RooRealVar("m_ditrack", "m_ditrack", cfg["mlo"], cfg["mhi"])
    mzv = ROOT.RooRealVar("fitted_mass", "fitted_mass", Z_LO, Z_HI)
    m  = np.ascontiguousarray(m,  dtype="float64")
    mz = np.ascontiguousarray(mz, dtype="float64")
    print("  building RooDataSet...", flush=True)
    data = ROOT.RooDataSet.from_numpy({"m_ditrack": m, "fitted_mass": mz}, [mtt, mzv])
    print("  dataset entries =", data.numEntries(), flush=True)

    model, sig, bkg, nsig, nbkg, shape, width = build_model(mtt, cfg, m.size)
    print("  model built; fitting...", flush=True)
    res = model.fitTo(data, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True),
                      ROOT.RooFit.PrintLevel(-1))
    print("  fit returned", flush=True)
    print(f"  fit status={res.status()}  Nsig={nsig.getVal():.0f}+/-{nsig.getError():.0f}"
          f"  Nbkg={nbkg.getVal():.0f}+/-{nbkg.getError():.0f}")
    draw_fit(label, mtt, data, model, sig, bkg, nsig, nbkg, cfg)

    # ---- SPlot: fix shapes, float only the yields ----
    for p in shape:
        p.setConstant(True)
    sData = ROOT.RooStats.SPlot("sData", "sData", data, model, ROOT.RooArgList(nsig, nbkg))
    print(f"  sum(sig sWeights)={sData.GetYieldFromSWeight('nsig'):.0f}"
          f"  sum(bkg sWeights)={sData.GetYieldFromSWeight('nbkg'):.0f}")

    n = data.numEntries()
    sw_sig = np.array([sData.GetSWeight(i, "nsig") for i in range(n)])
    sw_bkg = np.array([sData.GetSWeight(i, "nbkg") for i in range(n)])
    hR, hS, hB = draw_splot(label, mz, sw_sig, sw_bkg)

    fo = ROOT.TFile(f"{OUT_DIR}/{label}_splot.root", "RECREATE")
    for h in (hR, hS, hB): h.Write()
    fo.Close()
    print(f"  wrote {OUT_DIR}/{label}_{{ditrack_fit,splot_Zsignal,splot_panels}}.png (+ .root)")

if __name__ == "__main__":
    main()
