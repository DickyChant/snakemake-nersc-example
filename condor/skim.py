#!/usr/bin/env python3
"""
Per-file skimmer -- the reusable unit for both the interactive skim and the condor pipeline.

Reads ONE NanoAOD (local /eos path or an xrootd URL) and writes a compact float32 .npz with the
per-candidate columns the sPlot / bump-hunt / CATHODE studies need, after a loose window
preselection (m_ditrack in the phi builder window, fitted_mass in [70,110]). Tight analysis cuts
(m(ll), pt, svprob, real-phi) are applied later on the cache, so we skim once and iterate fast.

Standalone: needs only ROOT + numpy (cmsenv OR an LCG view on a condor worker), NOT the BPHNano
dictionaries -- the slim NanoAOD stores plain flat branches.

    python3 skim.py <input.root|xrootd-url> <collection> <output.npz>
"""
import sys, numpy as np, ROOT

ROOT.gROOT.SetBatch(True)

COLS = ["m_ditrack", "fitted_mass", "mll_fullfit", "ditrack_pt", "dilep_pt", "fitted_pt",
        "svprob", "l_xy_sig", "cos2D", "v_iso", "min_dr", "max_dr",
        "trk1_svip2d", "trk2_svip2d"]

def main():
    inp, coll, out = sys.argv[1], sys.argv[2], sys.argv[3]
    pt_min  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0     # ditrack (V) pT lower cut
    mll_max = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9     # dilepton-mass upper cut
    br = [f"{coll}_{c}" for c in COLS]
    d = ROOT.RDataFrame("Events", inp).Filter(f"n{coll}>0")
    cols = d.AsNumpy(br)

    def flat(b):
        a = cols[b]
        return (np.concatenate([np.asarray(v, "f4") for v in a]) if len(a)
                else np.zeros(0, "f4"))
    arrs = {c: flat(f"{coll}_{c}") for c in COLS}
    m, z = arrs["m_ditrack"], arrs["fitted_mass"]
    sel = ((m > 1.00) & (m < 1.05) & (z > 70) & (z < 110)          # loose phi window + fitted-mass range
           & (arrs["ditrack_pt"] > pt_min) & (arrs["mll_fullfit"] < mll_max))
    np.savez_compressed(out, **{c: arrs[c][sel] for c in COLS})
    print(f"wrote {out}  ncand={int(sel.sum())}  (pt>{pt_min:g}, mll<{mll_max:g})")

if __name__ == "__main__":
    main()
