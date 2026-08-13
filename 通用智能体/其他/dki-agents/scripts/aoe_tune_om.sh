#!/usr/bin/env bash
# AOE（Ascend Optimization Engine）自动调优：对 NER ONNX 先子图调优(SGAT, job_type=1)
# 再算子调优(OPAT, job_type=2)，把最优策略固化到自定义知识库(TUNE_BANK_PATH)，
# 生成调优后 OM；随后由 acl_om_pipeline_bench 实测调优前/后吞吐对比。
#
# 用法（目标机）：
#   bash scripts/aoe_tune_om.sh <onnx_path> <out_prefix> <soc_version>
# 例：bash scripts/aoe_tune_om.sh outputs/bench/npu_int8/ner_ft_bs64_seq256_fp32.onnx \
#         outputs/bench/npu_aoe/ner_aoe Ascend910B2C
set -eo pipefail

ONNX="${1:?need onnx path}"
OUT="${2:?need output prefix}"
SOC="${3:-Ascend910B2C}"
SHAPE="${INPUT_SHAPE:-input_ids:64,256;attention_mask:64,256;token_type_ids:64,256}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export TUNE_BANK_PATH="${TUNE_BANK_PATH:-/root/aoe_knowledge_bank}"
mkdir -p "$TUNE_BANK_PATH" "$(dirname "$OUT")"

echo "== [1/3] baseline ATC（无 AOE 知识库）=="
atc --framework=5 --model="$ONNX" --output="${OUT}_baseline" \
    --input_format=ND --input_shape="$SHAPE" --soc_version="$SOC" 2>&1 | tail -5

echo "== [2/3] AOE 子图调优 SGAT（job_type=1）=="
aoe --framework=5 --model="$ONNX" --job_type=1 \
    --input_format=ND --input_shape="$SHAPE" --soc_version="$SOC" \
    --tune_bank_path="$TUNE_BANK_PATH" 2>&1 | tail -8 || echo "SGAT 结束（部分算子可能不支持，属正常）"

echo "== [2/3] AOE 算子调优 OPAT（job_type=2）=="
aoe --framework=5 --model="$ONNX" --job_type=2 \
    --input_format=ND --input_shape="$SHAPE" --soc_version="$SOC" \
    --tune_bank_path="$TUNE_BANK_PATH" 2>&1 | tail -8 || echo "OPAT 结束"

echo "== [3/3] 用 AOE 知识库重新 ATC 生成调优后 OM =="
# ATC 会自动读取 TUNE_BANK_PATH 中的自定义知识库
atc --framework=5 --model="$ONNX" --output="${OUT}_aoe" \
    --input_format=ND --input_shape="$SHAPE" --soc_version="$SOC" 2>&1 | tail -5

echo "AOE_TUNE_DONE baseline=${OUT}_baseline.om aoe=${OUT}_aoe.om bank=$TUNE_BANK_PATH"
ls -la "${OUT}_baseline.om" "${OUT}_aoe.om" 2>/dev/null || true
