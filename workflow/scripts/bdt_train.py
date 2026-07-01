#!/usr/bin/env python3
"""
Supervised BDT for Z -> mu mu phi:  signal MC (real-phi)  vs  data background (the skim cache).

The data is ~pure background (the bump hunt gave 0 sigma), so signal-MC-vs-data teaches the BDT
signal-like vs background-like kinematics. The resonant variable m(ll phi)=fitted_mass is EXCLUDED
from the inputs so (a) the BDT can't cheat on the mass and (b) we can still bump-hunt m(ll phi)
after a BDT cut. Reports AUC, feature importance, the score distributions, and the post-BDT
m(ll phi) spectrum. NB: mll is correlated with m(ll phi), so a tight BDT cut can sculpt the mass
shape -- a decorrelated-feature version is the natural follow-up. genWeight is absent in the
private signal, so signal is unit-weighted (shapes only).

    python3 bdt_train.py                 # ZToMuMuPhi, all signal files vs cache background
"""
import glob, numpy as np, ROOT
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb

ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)
SIG_DIR = "/uscms_data/d3/sitianq/zllv/private_signals"
CACHE   = "/uscms_data/d3/sitianq/zllv/skim_cache/Muon0_ZToMuMuPhi.npz"
OUT_DIR = "/uscms_data/d3/sitianq/zllv/plots"
COLL, SIGGLOB = "ZToMuMuPhi", "ZToPhimumu_*.root"
PHI = (1.008, 1.032)
FEATS = ["mll_fullfit", "ditrack_pt", "dilep_pt", "fitted_pt", "svprob", "l_xy_sig",
         "cos2D", "v_iso", "min_dr", "max_dr", "trk1_svip2d", "trk2_svip2d"]

def load_signal(cap=200000):
    br = [f"{COLL}_m_ditrack", f"{COLL}_fitted_mass"] + [f"{COLL}_{f}" for f in FEATS]
    acc = {b: [] for b in br}; n = 0
    for fpath in sorted(glob.glob(f"{SIG_DIR}/{SIGGLOB}")):
        d = ROOT.RDataFrame("Events", fpath).Filter(f"n{COLL}>0")
        c = d.AsNumpy(br)
        if len(c[br[0]]) == 0:
            continue
        flat = {b: np.concatenate([np.asarray(v, "f8") for v in c[b]]) for b in br}
        m, z = flat[br[0]], flat[br[1]]
        sel = (m > PHI[0]) & (m < PHI[1]) & (z > 70) & (z < 110)
        for b in br:
            acc[b].append(flat[b][sel])
        n += int(sel.sum())
        if n >= cap:
            break
    out = {b: np.concatenate(a) for b, a in acc.items()}
    X = np.column_stack([out[f"{COLL}_{f}"] for f in FEATS])
    return X, out[f"{COLL}_fitted_mass"]

def load_bkg(cap=200000):
    d = np.load(CACHE)
    m, z = d["m_ditrack"], d["fitted_mass"]
    sel = (m > PHI[0]) & (m < PHI[1]) & (z > 70) & (z < 110)
    idx = np.where(sel)[0]
    rng = np.random.default_rng(0)
    if idx.size > cap:
        idx = rng.choice(idx, cap, replace=False)
    X = np.column_stack([d[f][idx] for f in FEATS])
    return X, z[idx]

def main():
    import os; os.makedirs(OUT_DIR, exist_ok=True)
    Xs, zs = load_signal()
    Xb, zb = load_bkg(cap=max(2 * len(Xs), 100000))
    print(f"signal (real-phi MC): {len(Xs)}   background (data cache): {len(Xb)}")

    X = np.vstack([Xs, Xb]); y = np.concatenate([np.ones(len(Xs)), np.zeros(len(Xb))])
    zmass = np.concatenate([zs, zb])
    Xtr, Xte, ytr, yte, ztr, zte = train_test_split(X, y, zmass, test_size=0.4, random_state=1, stratify=y)
    spw = (y == 0).sum() / (y == 1).sum()
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                            scale_pos_weight=spw, n_jobs=4)
    clf.fit(Xtr, ytr)
    ptr, pte = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xte)[:, 1]
    auc_tr, auc_te = roc_auc_score(ytr, ptr), roc_auc_score(yte, pte)
    print(f"AUC train={auc_tr:.3f}  test={auc_te:.3f}")
    imp = sorted(zip(FEATS, clf.feature_importances_), key=lambda t: -t[1])
    print("feature importance:")
    for f, v in imp:
        print(f"  {f:14s} {v:.3f}")

    # data-background m(llphi) spectrum vs increasing BDT cut (uses the TEST background only)
    bkg_te = yte == 0
    zb_te, pb_te = zte[bkg_te], pte[bkg_te]

    c = ROOT.TCanvas("c", "", 1300, 900); c.Divide(2, 2); keep = []
    # (1) ROC
    c.cd(1)
    fpr, tpr, _ = roc_curve(yte, pte)
    g = ROOT.TGraph(len(fpr), np.ascontiguousarray(fpr), np.ascontiguousarray(tpr))
    g.SetTitle(""); g.SetLineColor(ROOT.kBlue); g.SetLineWidth(3)
    g.GetXaxis().SetTitle("background eff (data)"); g.GetYaxis().SetTitle("signal eff (MC)")
    g.Draw("AL")
    dia = ROOT.TGraph(2); dia.SetPoint(0, 0, 0); dia.SetPoint(1, 1, 1); dia.SetLineStyle(2); dia.Draw("L")
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.045)
    t.DrawLatex(0.4, 0.3, f"AUC = {auc_te:.3f}"); keep += [g, dia, t]
    # (2) feature importance
    c.cd(2); ROOT.gPad.SetLeftMargin(0.28)
    hb = ROOT.TH1D("imp", ";relative importance;", len(imp), 0, len(imp))
    for i, (f, v) in enumerate(reversed(imp), 1):
        hb.SetBinContent(i, v); hb.GetXaxis().SetBinLabel(i, f)
    hb.SetFillColor(ROOT.kAzure + 1); hb.SetBarWidth(0.8); hb.SetBarOffset(0.1)
    hb.GetXaxis().SetLabelSize(0.05); hb.Draw("hbar"); keep.append(hb)
    tt = ROOT.TLatex(); tt.SetNDC(); tt.SetTextSize(0.045); tt.DrawLatex(0.3, 0.93, "feature importance"); keep.append(tt)
    # (3) BDT score: signal vs background (test)
    c.cd(3); ROOT.gPad.SetLogy()
    hs = ROOT.TH1D("hs", ";BDT score;a.u.", 40, 0, 1); hd = ROOT.TH1D("hd", "", 40, 0, 1)
    for v in pte[yte == 1]: hs.Fill(v)
    for v in pte[yte == 0]: hd.Fill(v)
    for h, col in ((hs, ROOT.kRed + 1), (hd, ROOT.kBlack)):
        if h.Integral() > 0: h.Scale(1.0 / h.Integral())
        h.SetLineColor(col); h.SetLineWidth(2)
    hs.SetMaximum(5 * max(hs.GetMaximum(), hd.GetMaximum())); hs.Draw("hist"); hd.Draw("hist same")
    leg = ROOT.TLegend(0.4, 0.75, 0.88, 0.88); leg.AddEntry(hs, "signal MC", "l"); leg.AddEntry(hd, "data (bkg)", "l"); leg.Draw()
    keep += [hs, hd, leg]
    # (4) post-BDT data m(llphi) at increasing score cuts (bump hunt sanity)
    c.cd(4)
    cols = [ROOT.kGray + 2, ROOT.kAzure + 1, ROOT.kOrange + 7, ROOT.kRed + 1]
    leg4 = ROOT.TLegend(0.5, 0.68, 0.88, 0.88); first = True
    for j, q in enumerate([0.0, 0.8, 0.95, 0.99]):
        thr = np.quantile(pb_te, q) if q > 0 else -1
        zz = zb_te[pb_te >= thr]
        h = ROOT.TH1D(f"z{j}", ";m(ll#phi) [GeV];norm.", 40, 70, 110)
        for v in zz: h.Fill(v)
        if h.Integral() > 0: h.Scale(1.0 / h.Integral())
        h.SetLineColor(cols[j]); h.SetLineWidth(2)
        h.SetMaximum(0.09); h.SetMinimum(0)
        h.Draw("hist" if first else "hist same"); first = False
        leg4.AddEntry(h, f"keep top {int((1-q)*100)}%", "l"); keep.append(h)
    leg4.Draw(); keep.append(leg4)
    l91 = ROOT.TLine(91, 0, 91, 0.09); l91.SetLineStyle(2); l91.SetLineColor(ROOT.kGreen + 2); l91.Draw(); keep.append(l91)
    c.cd(0); th = ROOT.TLatex(); th.SetNDC(); th.SetTextSize(0.025)
    th.DrawLatex(0.30, 0.975, "ZToMuMuPhi BDT: signal MC vs data background"); keep.append(th)
    out = f"{OUT_DIR}/ZToMuMuPhi_bdt.png"
    c.SaveAs(out); print(f"wrote {out}")

if __name__ == "__main__":
    main()
