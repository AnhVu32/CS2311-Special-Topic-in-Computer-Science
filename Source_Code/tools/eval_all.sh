#!/bin/bash
# Chấm mọi file *_samples.jsonl dưới một thư mục bằng EvalPlus (qua run_benchmark evaluate).
# Bỏ qua thư mục _invalid/ và các file đệm *_padded.jsonl. File chưa đủ 378 bài / còn stub sẽ bị từ chối.
#
#   bash Source_Code/tools/eval_all.sh                                   # mặc định Data/results/benchmark
#   bash Source_Code/tools/eval_all.sh Data/results/benchmark --evaluator docker
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"            # .../Source_Code
ROOT="${1:-$HERE/../Data/results/benchmark}"
[ $# -gt 0 ] && shift
ROOT="$(cd "$ROOT" && pwd)"
cd "$HERE"
find "$ROOT" -name "*_samples.jsonl" -not -path "*/_invalid/*" | sort | while read -r f; do
  echo "=== $f"
  python -m agent0.scripts.run_benchmark evaluate --samples_file "$f" "$@" || echo "   -> bỏ qua (chưa đủ bài hoặc lỗi chấm)"
done
echo "XONG $(date +%T)"
