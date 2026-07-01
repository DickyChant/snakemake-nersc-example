#!/bin/bash
# HTCondor worker wrapper: skim ONE file (picked from filelist.txt by job index) using a
# standalone LCG ROOT+numpy view from cvmfs -- no cmsenv / no /uscms_data dependency. The input
# is read over xrootd (needs the shipped grid proxy); the output .npz is transferred back.
#
#   arguments = $(Process) <collection>
set -e
IDX=$1; COLL=$2
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh
FILE=$(sed -n "$((IDX + 1))p" filelist.txt)
echo "job $IDX: $FILE"
python3 skim.py "$FILE" "$COLL" "skim_$(printf '%03d' "$IDX").npz"
