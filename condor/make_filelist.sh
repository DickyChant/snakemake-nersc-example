#!/bin/bash
# Write filelist.txt (xrootd URLs, one per line) for a primary dataset, and print the count so
# you can set `queue N` in submit.jdl.
#     bash make_filelist.sh [PD=Muon0]
PD=${1:-Muon0}
BASE=/eos/uscms/store/user/sitianq/zllv/ZLLV_zllv_2026Jun19
: > filelist.txt
for F in $BASE/$PD/crab_*/*/0000/zllv_nano_*.root; do
  echo "root://cmseos.fnal.gov/${F#/eos/uscms}" >> filelist.txt
done
echo "filelist.txt: $(wc -l < filelist.txt) files for $PD  (set 'queue N' in submit.jdl to this)"
