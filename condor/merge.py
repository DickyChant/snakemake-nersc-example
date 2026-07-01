#!/usr/bin/env python3
"""
Merge the per-file skim .npz (from skim_local.sh or the condor jobs) into one cache .npz that
the sPlot / bump-hunt / CATHODE scripts load instead of re-reading EOS.

    python3 merge.py <skim_dir> <out.npz>
"""
import sys, glob, numpy as np

def main():
    skim_dir, out = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(f"{skim_dir}/skim_*.npz"))
    if not files:
        sys.exit(f"no skim_*.npz in {skim_dir}")
    keys = list(np.load(files[0]).keys())
    acc = {k: [] for k in keys}
    for f in files:
        d = np.load(f)
        for k in keys:
            acc[k].append(d[k])
    merged = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(out, **merged)
    print(f"merged {len(files)} files -> {out}  ncand={merged[keys[0]].size}")

if __name__ == "__main__":
    main()
