# aa token prediction v.s. structure token prediction

This repository contains the reproduction scripts for the amino-acid and structure-token training runs.

## Setup

Use the same environment and dataset preparation described in `https://github.com/SuperCarryDFY/b097bade29ef6c984ea1c6f28ec5ece165c3765a459f16085d618a97f9a60286#installation`.
For dataset preparation, downloading `datasets_mini` from the download section is sufficient.

Before running the scripts, create `.env` in this repository and set the local paths:

```bash
PROJECT_ROOT=/path/to/this/repo
DATA_ROOT=/path/to/datasets
HF_MODELS_ROOT=hf_models
CACHE_ROOT=/path/to/cache
OUTPUT_ROOT=${PROJECT_ROOT}/output
CONDA_NAME=~/miniconda3
ENV_NAME=pinal
```

The code downloads standard HuggingFace models to `HF_MODELS_ROOT` when they are not found locally:

```text
google/flan-t5-base
gpt2
westlake-repl/SaProt_650M_AF2
```

## Reproduce

Run the full workflow from the repository root:

```bash
bash scripts/run_all.sh
```

This script runs:

1. Structure-token training: `scripts/train/structure_seqs.sh`
2. Amino-acid-token training: `scripts/train/aa_seqs.sh`
3. Sampling for both trained models with `scripts/generation/generate.sh`
4. ESMFold evaluation for both models with `scripts/eval/esmfold.sh`

The expected training output directories are:

```text
output/StructureTokenPrediction
output/AATokenPrediction
```

During training, train and validation next token prediction accuracy are logged automatically with Weights & Biases and TensorBoard. You can also plot the next token prediction accuracy curves from training logs:

```bash
python scripts/plot_acc_curves.py \
  --aa-log output/AATokenPrediction/out.log \
  --structure-log output/StructureTokenPrediction/out.log
```

After ESMFold evaluation, you can plot the pLDDT curves:

```bash
python scripts/plot_plddt_curves.py \
  --aa-dir output/AATokenPrediction \
  --structure-dir output/StructureTokenPrediction
```

Generated FASTA files are saved under:

```text
output/<experiment>/IntervalCheckpoints/step=<step>/sample_SwissProt-hard-500/seq.fasta
```

ESMFold outputs are saved next to each FASTA file as `seq_esmfold_results/` and `seq_esmfold_results.json`.
