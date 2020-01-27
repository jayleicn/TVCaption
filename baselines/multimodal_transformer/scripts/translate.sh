#!/usr/bin/env bash
# Usage:  bash baselines/multimodal_transformer/scripts/translate.sh res_2019_11_12_23_36_36 val
res_dir=$1
split_name=$2
results_root=baselines/multimodal_transformer/results

eval_path=data/tvc_${split_name}_release.jsonl
extra_args=()
if [[ ${split_name} == val ]]; then
    reference_path=data/tvc_${split_name}_release.jsonl
    extra_args+=(-reference_path)
    extra_args+=(${reference_path})
fi

python baselines/multimodal_transformer/translate.py \
-res_dir=${results_root}/${res_dir} \
-eval_split_name=${split_name} \
-eval_path=${eval_path} \
${extra_args[@]} \
${@:3}
