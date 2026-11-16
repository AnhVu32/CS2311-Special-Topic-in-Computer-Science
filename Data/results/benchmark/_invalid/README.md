# Các lần chạy không dùng để báo cáo

Giữ lại để đối chiếu. Lý do:

| File | Vấn đề |
|---|---|
| `gemma3_12b/baseline_samples.jsonl` | 245/378 bài là stub (lỗi API bị đổi thành `def f(*args, **kwargs): pass`) |
| `gemma3_12b/with_kb_n1_samples.jsonl` | 346/378 stub |
| `gemma3_12b/with_kb_n3_samples.jsonl` | 320/378 stub |
| `gemma3_12b/with_kb_n5_samples.jsonl` | 335/378 stub |
| `gemma3_12b/with_kb_samples.jsonl` | 63/378 stub |
| `gemma3_4b/baseline_samples.jsonl` | 358/378 stub |
| `llama3.1_8b/baseline_samples.jsonl` | 328/378 stub |
| `ministral-3_8b/with_kb_n4_samples.jsonl` | 50/378 stub |
| `ministral-3_8b/with_kb_n5_samples.jsonl` | 41/378 stub |
| `gemma3_4b/*_samples_eval_results.json` | chấm trên macOS: 378/378 bài "timeout" ở cả base và plus |
| `<model>/benchmark_summary.json` | số liệu của bộ chấm cũ: chạy trên macOS (MBPP+ thấp hơn thực tế 17–20 điểm) và chỉ tính trên các bài không phải stub, nên mỗi chế độ được chấm trên một tập bài khác nhau |

Ngưỡng tách: > 5% số bài (≥ 19/378) là stub. Các file còn 1–2 stub vẫn để trong `benchmark/` (ghi chú ở `rescored_linux/`).
Kết quả chấm lại đúng cách của mọi file (kể cả các file này): `../../rescored_linux/`.
