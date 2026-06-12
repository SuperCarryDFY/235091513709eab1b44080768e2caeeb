set -e

bash scripts/train/structure_seqs.sh
bash scripts/train/aa_seqs.sh

## eval 
bash scripts/generation/generate.sh configs/train/StructureTokenPrediction.yaml output/StructureTokenPrediction
bash scripts/generation/generate.sh configs/train/AATokenPrediction.yaml output/AATokenPrediction

## esmfold 
bash scripts/eval/esmfold.sh output/StructureTokenPrediction
bash scripts/eval/esmfold.sh output/AATokenPrediction

