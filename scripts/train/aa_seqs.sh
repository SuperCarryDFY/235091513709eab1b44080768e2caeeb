#! /bin/bash 

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi
cd "$PROJECT_ROOT"

CONDA_NAME=${CONDA_NAME:-$(conda info --base)}
ENV_NAME=${ENV_NAME:-genesis}
source "$CONDA_NAME/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export NUM_NODES=1
accelerate launch \
  --config_file accelerate_config/deepspeed.yaml\
  experiments/train.py \
  --config configs/train/AATokenPrediction.yaml
