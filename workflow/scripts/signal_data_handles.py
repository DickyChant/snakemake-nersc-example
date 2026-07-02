#!/usr/bin/env python3
"""Signal-MC vs data discriminating handles (m(ll), pT(phi), Iso(phi)) + the m(ll)-vs-m(llphi)
search plane, inclusive vs after the harder cuts. Reads the loose skim cache (data) + signal MC.
Real-phi window. See also dalitz_signal.py."""
ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)
_KEEP = []
OUT="/uscms_data/d3/sitianq/zllv/plots"; coll="ZToMuMuPhi"; PHI=(1.008,1.032)
V=["mll_fullfit","ditrack_pt","v_iso","fitted_mass"]

def load_sig(files):
    acc={v:[] for v in V}
    for fn in files:
        d=ROOT.RDataFrame("Events",fn).Filter(f"n{coll}>0")
        c=d.AsNumpy([f"{coll}_m_ditrack"]+[f"{coll}_{v}" for v in V])
        if len(c[f"{coll}_m_ditrack"])==0: continue
        m=np.concatenate([np.asarray(x,'f8') for x in c[f"{coll}_m_ditrack"]])
        w=(m>PHI[0])&(m<PHI[1])
        for v in V: acc[v].append(np.concatenate([np.asarray(x,'f8') for x in c[f"{coll}_{v}"]])[w])
    return {v:np.concatenate(a) for v,a in acc.items()}

S=load_sig(sorted(glob.glob("/uscms_data/d3/sitianq/zllv/private_signals/ZToPhimumu_*.root"))[:25])
dd=np.load("/uscms_data/d3/sitianq/zllv/skim_cache/Muon0_ZToMuMuPhi.npz")
w=(dd['m_ditrack']>PHI[0])&(dd['m_ditrack']<PHI[1])
D={v:dd[v][w] for v in V}
MLL,PT=70.0,10.0; ISO=float(np.percentile(S['v_iso'],85))
print("iso cut (85pct sig): %.3f"%ISO)

def overlay(pad,s,d,nb,lo,hi,xt,cutx):
    pad.cd()
    hs=ROOT.TH1D("s"+xt,"",nb,lo,hi); hd=ROOT.TH1D("d"+xt,"",nb,lo,hi)
    for x in s: hs.Fill(x)
    for x in d: hd.Fill(x)
    for h,col in ((hs,ROOT.kAzure+1),(hd,ROOT.kBlack)):
        if h.Integral()>0: h.Scale(1/h.Integral())
        h.SetLineColor(col); h.SetLineWidth(2)
    hd.SetMarkerStyle(20); hd.SetMarkerSize(0.5)
    hs.SetMaximum(1.3*max(hs.GetMaximum(),hd.GetMaximum())); hs.SetMinimum(0)
    hs.GetXaxis().SetTitle(xt); hs.SetFillColorAlpha(ROOT.kAzure+1,0.25)
    hs.Draw("hist"); hd.Draw("e same")
    ln=ROOT.TLine(cutx,0,cutx,hs.GetMaximum()); ln.SetLineColor(ROOT.kRed); ln.SetLineWidth(2); ln.Draw()
    _KEEP.extend([hs,hd,ln])

def plane(pad,L,Z,title):
    pad.cd(); ROOT.gPad.SetRightMargin(0.14); ROOT.gPad.SetLogz()
    h=ROOT.TH2D("p"+title,title+";m(#mu#mu);m(#mu#mu#phi)",60,0,120,50,70,110)
    for a,b in zip(L,Z): h.Fill(a,b)
    h.Draw("colz")
    ln=ROOT.TLine(0,91.19,120,91.19); ln.SetLineColor(ROOT.kRed); ln.SetLineStyle(2); ln.Draw()
    _KEEP.extend([h,ln])

c=ROOT.TCanvas("c","",1500,900); c.Divide(3,2)
overlay(c.cd(1),S['mll_fullfit'],D['mll_fullfit'],60,0,120,"m(#mu#mu) [GeV]",MLL)
overlay(c.cd(2),S['ditrack_pt'],D['ditrack_pt'],50,0,50,"p_{T}(#phi) [GeV]",PT)
overlay(c.cd(3),S['v_iso'],D['v_iso'],50,0,float(np.percentile(D['v_iso'],99)),"Iso(#phi) v_iso",ISO)
plane(c.cd(4),D['mll_fullfit'][:1500000],D['fitted_mass'][:1500000],"data inclusive")
dc=(D['ditrack_pt']>PT)&(D['v_iso']<ISO)&(D['mll_fullfit']<MLL)
sc=(S['ditrack_pt']>PT)&(S['v_iso']<ISO)&(S['mll_fullfit']<MLL)
plane(c.cd(5),D['mll_fullfit'][dc],D['fitted_mass'][dc],"data | pT>10,iso<%.1f,mll<70"%ISO)
plane(c.cd(6),S['mll_fullfit'][sc],S['fitted_mass'][sc],"signal | same cuts")
T=ROOT.TLatex(); T.SetNDC(); T.SetTextSize(0.02); c.cd(0)
T.DrawLatex(0.18,0.985,"top: handles (blue=sig, black=data, red=cut)   bottom: m(mumu) vs m(mumuphi)  inclusive / after harder cuts / signal"); _KEEP.append(T)
print("data after cuts: %d of %d ; signal eff: %.3f"%(int(dc.sum()),dc.size,sc.mean()))
c.SaveAs(f"{OUT}/ZToMuMuPhi_handles.png"); print("wrote",f"{OUT}/ZToMuMuPhi_handles.png")
