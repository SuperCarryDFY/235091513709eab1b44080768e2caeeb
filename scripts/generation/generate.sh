set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/output}

EVAL_STEPS="5000 10000 15000 20000"

CONDA_NAME=${CONDA_NAME:-$(conda info --base)}
ENV_NAME=${ENV_NAME:-genesis}
source "$CONDA_NAME/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"

CONFIG_PATH=$1
EXP_DIR=$2

for EVAL_STEP in $EVAL_STEPS; do
    cd "$EXP_DIR/IntervalCheckpoints/step=$EVAL_STEP"
    python zero_to_fp32.py . pytorch_model.bin
    cd "$PROJECT_ROOT"
    accelerate launch experiments/sample.py \
        --config $CONFIG_PATH \
        --ckpt_path $EXP_DIR/IntervalCheckpoints/step\=$EVAL_STEP/pytorch_model.bin \
        --sample_config configs/sample/SwissProt-hard-500.yaml

done
