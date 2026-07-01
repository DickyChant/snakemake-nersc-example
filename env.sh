#!/usr/bin/env bash
# env.sh -- enter the CMSSW runtime (PyROOT + RooFit + RooStats), then exec the given command.
#
# The zllv analysis scripts use PyROOT/RooFit from the CMSSW_15_0_15 release, so every
# Snakemake rule wraps its command with this so the job runs inside `cmsenv` regardless of
# executor (local shell or an HTCondor worker). Adjust CMSSW_BASE if the release moves.
set -euo pipefail
export SCRAM_ARCH=el9_amd64_gcc12
CMSSW_SRC=/uscms_data/d3/sitianq/bph_nano/CMSSW_15_0_15/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd "$CMSSW_SRC" >/dev/null
eval "$(scramv1 runtime -sh)"
popd >/dev/null
exec "$@"
