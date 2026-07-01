# Snakemake with SLURM on the CMS Tier-3 at PSI

> **`lpc` branch — zllv analysis on the FNAL LPC.** This branch reuses the upstream
> pixi + Snakemake scaffolding but replaces the toy generate→gather→sort demo with the real
> **Z → ℓℓ V, V → h⁺h⁻** analysis, and targets the LPC (HTCondor, `/eos/uscms`) instead of
> PSI/SLURM. See **[zllv workflow](#zllv-workflow-lpc-branch)** below. The original PSI demo
> stages and tasks are preserved for reference.
>
> ## zllv workflow (LPC branch)
>
> Two stages, wired in [`workflow/Snakefile`](workflow/Snakefile), each wrapped in
> [`env.sh`](env.sh) so it runs inside CMSSW `cmsenv` (PyROOT/RooFit) on any node:
>
> | stage | rule | script | outputs (in `plots/`) |
> |-------|------|--------|-----------------------|
> | A | `signal_efficiency` | `../signal_efficiency.py` | `eff_signal.png` (selection-efficiency cutflow over the private-signal NanoAODs) |
> | B | `data_splot` (per collection) | `../data_splot.py` | `<coll>_ditrack_fit.png`, `<coll>_splot_Zsignal.png`, `<coll>_splot_panels.png` (fit m(hh) → sWeights → background-subtracted m(ℓℓV)) |
>
> Paths, the signal-process / data-collection lists, and the sPlot caps live in
> [`config.yaml`](config.yaml). Set up pixi, then:
>
> ```shell
> source /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/latest/setup.sh
> pixi install
> pixi run zllv_dry     # preview the DAG
> pixi run zllv         # run locally on the interactive node (profiles/lpc)
> pixi run zllv_condor  # optional: fan out over HTCondor (profiles/lpc-condor)
> ```
>
> The local profile is the tested path; the HTCondor profile in `profiles/lpc-condor/` is a
> documented starting point (validate the condor submit string against your quota first).
>
> ---
>
> ## Upstream PSI/SLURM demo

This repository demonstrates a Snakemake workflow with three stages:
1. **Generate** - Create random numbers in parallel jobs
2. **Gather** - Combine all generated files
3. **Sort** - Sort the combined numbers

Clone this repository. Set up `pixi`:

```shell
source /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/latest/setup.sh
```

Run tasks:

```shell
# Local execution
pixi run local

# SLURM execution
pixi run slurm

# SLURM with Apptainer containers
pixi run slurm_apptainer

# Local with Apptainer containers
pixi run apptainer

# Local execution with XRootD via EOS
pixi run xrootd

# SLURM execution with XRootD via EOS
pixi run slurm_xrootd

# SLURM with Apptainer containers and XRootD via EOS
pixi run slurm_apptainer_xrootd

# Local with Apptainer containers and XRootD via EOS
pixi run apptainer_xrootd
```

## How to start from scratch

Set up `pixi` as shown above. Then initialise the workspace:

```shell
pixi init snakemake-psi-tier3-example
cd snakemake-psi-tier3-example
pixi workspace channel add bioconda
pixi add snakemake
pixi add snakemake-executor-plugin-slurm
```

### Cluster-specific settings

Generate partition file (**this has to be done for each cluster**):

```shell
generate-slurm-partition-config -o partitions.yaml
```

Please also mind that the `profiles/config.yaml` needs to be adjusted to the queue
names and slurm user of the given cluster:

```txt
default-resources:
  slurm_partition: short  # Default SLURM partition
  slurm_account: t3       # SLURM account for job submission
```

### Testing

To test the local workflow:

```shell
pixi shell
snakemake --verbose --cores 1 --printshellcmds --snakefile workflow/Snakefile sort_script
```

To test the SLURM workflow (assuming you're still in the `pixi shell`):

```shell
rm results/sorted_numbers_script.txt
export SNAKEMAKE_SLURM_PARTITIONS=${PWD}/partitions.yaml
snakemake --executor slurm \
  --workflow-profile ${PWD}/profiles \
  -j unlimited \
  --verbose \
  --printshellcmds \
  --snakefile workflow/Snakefile sort_script
```

### XRootD / EOS integration

To write workflow outputs directly to EOS via XRootD, you need a valid CERN Kerberos token in your environment (e.g., pointed to by `$KRB5CCNAME`). XRootD-enabled targets can be triggered via:

```shell
pixi run xrootd
pixi run slurm_xrootd
pixi run slurm_apptainer_xrootd
pixi run apptainer_xrootd
```

These commands use Snakemake's `--default-storage-provider` to transparently map all inputs and outputs to your XRootD storage prefix without having to modify the `workflow/Snakefile`'s logic. You can adjust the destination path inside `pixi.toml` under the XRootD tasks.
The setting of `env XrdSecPROTOCOL=krb5,unix` enforces the use of the Kerberos token instead of using GSI authenticiation (grid certificate).

## Configuration

The workflow parameters can be adjusted in `config.yaml`:
- `num_files`: Number of parallel jobs to generate random numbers (default: 5)
- `numbers_per_file`: Random numbers generated per file (default: 10)

SLURM executor settings are in `profiles/config.yaml`.

## Workflow Structure

```
generate_numbers (parallel)
    ↓
    data/generated/numbers_0.txt
    data/generated/numbers_1.txt
    ...
    data/generated/numbers_N.txt
    ↓
gather_numbers
    ↓
    data/numbers_gathered.txt
    ↓
sort_script (or sort_local)
    ↓
    results/sorted_numbers_script.txt
```

## Cleaning Up

To remove generated files and start fresh:
```shell
rm -rf results/ data/generated/ data/numbers_gathered.txt .snakemake/
```
