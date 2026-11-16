# Chấm lại toàn bộ bằng EvalPlus trên Linux

- Ngày chấm: 15/11/2026. EvalPlus 0.3.1, MBPP+ v0.2.0 (378 bài), Linux x86_64, `--parallel 2`.
- Đầu vào: 35 file `*_samples.jsonl` của dự án (26 file hợp lệ trong `../benchmark/`, 9 file hỏng trong `../benchmark/_invalid/`).
- Chấm lại lần hai 9 file với `--parallel 1`: 3 400/3 402 kết quả từng bài giống hệt (chỉ Mbpp/160 đổi ở 2 file) → kết quả ổn định.

| File | Nội dung |
|---|---|
| `rescored_summary.csv` | pass@1 base / plus của từng lần chạy: trên đủ 378 bài (stub = fail) và trên riêng các bài không phải stub; số stub |
| `paired_vs_baseline.csv` | so sánh cặp với baseline cùng model: số bài chung, Δ (điểm %), b = baseline fail → KB pass, c = baseline pass → KB fail, p McNemar exact |
| `rescored.json` | kết quả pass/fail từng bài của cả 35 lần chạy |
| `eval_results_invalid_runs/` | file kết quả EvalPlus của 9 lần chạy hỏng (kết quả của 26 file hợp lệ nằm ngay cạnh file samples trong `../benchmark/<model>/`) |

Tái lập (từ thư mục `Source_Code`, trên Linux):

```bash
bash tools/eval_all.sh ../Data/results/benchmark --evaluator local
python tools/rescore_all.py ../Data/results/benchmark ../Data/results/rescored_linux
```

Lưu ý khi đọc bảng: 1 bài = 0.26 điểm %; baseline tham chiếu của gemma3:12b và gemma3:4b là `baseline_v2` (bản `baseline` đầu có 245 và 358 stub).
