set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

CONDA_NAME=${CONDA_NAME:-$(conda info --base)}
ENV_NAME=${ENV_NAME:-genesis}
source "$CONDA_NAME/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$PROJECT_ROOT"
EVAL_STEPS="0 5000 10000 15000 20000"

TEST_DIR=$1
for EVAL_STEP in $EVAL_STEPS; do
    FASTA_PATH=$TEST_DIR/IntervalCheckpoints/step=$EVAL_STEP/sample_SwissProt-hard-500/seq.fasta
    python analysis/esmfold/run.py --fasta_path $FASTA_PATH
done
