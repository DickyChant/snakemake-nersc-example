#!/usr/bin/env python3
"""
Gen-level Dalitz plot for the signal Z -> mu+ mu- phi (phi treated as one body).

The private ZToPhimumu sample decays Z(23) -> phi(333) mu-(13) mu+(-13) directly (the status-23
muons + status-2 phi all have the Z as mother; phi -> K+K-). We take those three gen particles,
build their 4-vectors, and fill the symmetric Dalitz m^2(mu+ phi) vs m^2(mu- phi). The dilepton
mass runs along the anti-diagonal: m^2(mu mu) = m_Z^2 + 2 m_mu^2 + m_phi^2 - m^2(mu+phi) - m^2(mu-phi),
so low m(mu mu) (the signal signature) sits toward the upper-right corner. A second panel shows the
m(mu mu) spectrum directly.

    python3 dalitz_signal.py                 # ~20 files
"""
import ROOT, glob, argparse, numpy as np

ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)
SIG_DIR = "/uscms_data/d3/sitianq/zllv/private_signals"
OUT_DIR = "/uscms_data/d3/sitianq/zllv/plots"
MU = 0.1056583745

def p4(pt, eta, phi, m):
    px = pt*np.cos(phi); py = pt*np.sin(phi); pz = pt*np.sinh(eta)
    return np.array([np.sqrt(px*px+py*py+pz*pz+m*m), px, py, pz])

def m2(a, b):
    s = a + b
    return s[0]**2 - s[1]**2 - s[2]**2 - s[3]**2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=20)
    ap.add_argument("--proc", default="ZToPhimumu")
    args = ap.parse_args()
    import os; os.makedirs(OUT_DIR, exist_ok=True)

    hD = ROOT.TH2D("dalitz", ";m^{2}(#mu^{+}#phi) [GeV^{2}];m^{2}(#mu^{-}#phi) [GeV^{2}]",
                   90, 0, 8600, 90, 0, 8600)
    hMll = ROOT.TH1D("mll", ";m(#mu#mu) [GeV];gen events", 60, 0, 95)
    nfill = 0
    for fn in sorted(glob.glob(f"{SIG_DIR}/{args.proc}_*.root"))[:args.files]:
        f = ROOT.TFile.Open(fn)
        if not f or f.IsZombie():
            continue
        t = f.Get("Events")
        for ev in range(t.GetEntries()):
            t.GetEntry(ev)
            n = t.nGenPart
            pid = t.GenPart_pdgId; mom = t.GenPart_genPartIdxMother; st = t.GenPart_status
            pt = t.GenPart_pt; eta = t.GenPart_eta; phi = t.GenPart_phi; mass = t.GenPart_mass
            iphi = imup = imum = -1
            for i in range(n):
                mp = pid[mom[i]] if 0 <= mom[i] < n else 0
                if abs(mp) != 23:
                    continue
                if pid[i] == 333:              iphi = i
                elif pid[i] == -13 and st[i] == 23: imup = i
                elif pid[i] == 13 and st[i] == 23:  imum = i
            if iphi < 0 or imup < 0 or imum < 0:
                continue
            vphi = p4(pt[iphi], eta[iphi], phi[iphi], mass[iphi])
            vp = p4(pt[imup], eta[imup], phi[imup], MU)
            vm = p4(pt[imum], eta[imum], phi[imum], MU)
            hD.Fill(m2(vp, vphi), m2(vm, vphi))
            hMll.Fill(np.sqrt(max(m2(vp, vm), 0)))
            nfill += 1
        f.Close()
    print(f"[{args.proc}] gen Z->mumu phi decays used: {nfill}")

    c = ROOT.TCanvas("c", "", 1200, 540); c.Divide(2, 1)
    c.cd(1); ROOT.gPad.SetRightMargin(0.14)
    hD.Draw("colz")
    t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(0.04)
    t.DrawLatex(0.12, 0.92, "Z#rightarrow#mu#mu#phi Dalitz (gen)")
    t.SetTextSize(0.03); t.SetTextAngle(38)
    t.DrawLatex(0.60, 0.62, "low m(#mu#mu) #rightarrow")
    c.cd(2)
    hMll.SetLineColor(ROOT.kAzure+1); hMll.SetLineWidth(2); hMll.SetMinimum(0)
    hMll.Draw("hist")
    t2 = ROOT.TLatex(); t2.SetNDC(); t2.SetTextSize(0.04)
    t2.DrawLatex(0.13, 0.92, "gen m(#mu#mu): pushed below m_{Z} by the #phi recoil")
    out = f"{OUT_DIR}/{args.proc}_dalitz.png"
    c.SaveAs(out); print(f"wrote {out}")

if __name__ == "__main__":
    main()
