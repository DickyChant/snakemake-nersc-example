#!/bin/bash
# Interactive sequential skim over ALL files of a primary dataset -> one per-file .npz each,
# in a persistent cache. RAM-safe (one file at a time). This is the "test interactive" path;
# the condor pipeline runs the same skim.py per file in parallel instead.
#
#   bash skim_local.sh [PD=Muon0] [COLL=ZToMuMuPhi] [OUTDIR]
set -u
export SCRAM_ARCH=el9_amd64_gcc12
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /uscms_data/d3/sitianq/bph_nano/CMSSW_15_0_15/src && eval "$(scramv1 runtime -sh)"
PD=${1:-Muon0}; COLL=${2:-ZToMuMuPhi}
OUTDIR=${3:-/uscms_data/d3/sitianq/zllv/skim_cache/$PD}
SKIM=/uscms_data/d3/sitianq/zllv/snakemake-nersc-example/condor/skim.py
mkdir -p "$OUTDIR"
i=0; ok=0
for F in /eos/uscms/store/user/sitianq/zllv/ZLLV_zllv_2026Jun19/$PD/crab_*/*/0000/zllv_nano_*.root; do
  if python3 "$SKIM" "$F" "$COLL" "$OUTDIR/skim_$(printf '%03d' $i).npz" >/dev/null 2>&1; then
    ok=$((ok+1))
  else
    echo "FAIL $i $F"
  fi
  i=$((i+1))
done
echo "SKIM ALL DONE: $ok/$i files ok -> $OUTDIR"
