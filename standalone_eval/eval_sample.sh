#!/usr/bin/env bash
# Usage:
#   bash standalone_eval/eval_sample.sh
submission_path=sample_val_predictions.jsonl
reference_path=../data/tvc_val_release.jsonl
output_path=sample_val_predictions_metrics.json

cd standalone_eval

python evaluate.py \
-s ${submission_path} \
-r ${reference_path} \
-o ${output_path}

cd ..
