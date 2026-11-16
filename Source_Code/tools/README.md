# tools

Chạy từ thư mục `Source_Code`.

| File | Việc | Lệnh |
|---|---|---|
| `eval_all.sh` | Chấm mọi `*_samples.jsonl` dưới một thư mục (bỏ qua `_invalid/`) qua `run_benchmark evaluate` | `bash tools/eval_all.sh ../Data/results/benchmark --evaluator docker` |
| `rescore_all.py` | Tổng hợp: pass@1 trên 378 bài + so sánh cặp với baseline cho mọi lần chạy → CSV | `python tools/rescore_all.py ../Data/results/benchmark ../Data/results/rescored_linux` |
| `check_kb.py` | Kiểm tra từng mục KB bằng harness chặt (nạp như module, đổi so sánh trần thành assert) | `python -I tools/check_kb.py ../Data/knowledge_base/all_models/knowledge_base.jsonl` |
| `clean_kb.py` | Kiểm lại KB bằng harness của pipeline, bỏ mục trùng đề / giống MBPP+, ghi KB đã lọc + báo cáo | `python tools/clean_kb.py ../Data/knowledge_base/all_models/knowledge_base.jsonl --out ../Data/knowledge_base/all_models_clean/knowledge_base.jsonl` |
| `check_contamination.py` | Độ trùng giữa KB và MBPP+ (TF-IDF, tên hàm, id bài MBPP) | `python tools/check_contamination.py ../Data/knowledge_base/all_models/knowledge_base.jsonl` |
| `run_matrix.py` | Benchmark nhiều model × cấu hình (baseline / retrieved / random / fixed), chạy tiếp được, ngân sách thời gian, chấm nền. `--exclude_test_repaired` bỏ mục KB có test bị sửa; `--kb_tag` để nhiều KB chung một thư mục kết quả; `--local_lanes 2` chạy hai model local cùng lúc trên máy hai GPU | `python tools/run_matrix.py --roster kaggle/roster.json --results_dir ../Data/results/matrix --exclude_test_repaired` |
| `summarize_matrix.py` | Bảng pass@1, so sánh cặp McNemar + Holm, Δ theo độ mạnh của model, biểu đồ; Δ theo độ giống của ví dụ truy hồi (độ liên quan / rò rỉ); so sánh giữa các KB; `--model_pairs A:B` so hai model trên cùng cấu hình (vd trước / sau GRPO) | `python tools/summarize_matrix.py --results_dir ../Data/results/matrix --roster kaggle/roster.json` |
| `split_kb_by_phat.py` | Tách KB đã gắn p̂ (`run_curriculum --p_hat_action tag`) thành KB `phat` (p̂ ∈ [0.25, 0.75]) và KB `ctrl` cùng cỡ, chọn ngẫu nhiên — đo tác dụng của việc chọn đề ở biên năng lực | `python tools/split_kb_by_phat.py <kb>.jsonl --out_dir ../Data/knowledge_base/split` |
| `grpo_executor.py` | RL cho executor kiểu Agent0 (GRPO + LoRA, TRL): `prepare` (đề KB → prompt MBPP+, lọc theo p̂ của model), `train` (thưởng = chạy test trong harness), `merge`, `score` (pass@1 trên val) | `python tools/grpo_executor.py prepare --kb <kb>.jsonl --out_dir <dir> --model Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| `hf_generate.py` | Sinh lời giải MBPP+ bằng transformers (model local / Hugging Face, kèm LoRA, có thể few-shot từ KB); kết quả chấm được bằng `run_benchmark evaluate` | `python tools/hf_generate.py --model <model> [--adapter <lora>] --output <dir>/<label>/baseline_samples.jsonl` |
| `package_for_kaggle.py` | Đóng gói project thành zip để tải lên Kaggle Datasets (không kèm `.env`) | `python3 tools/package_for_kaggle.py` |
| `convert_mbpp_to_parquet.py` | Đổi `mbpp.jsonl` sang parquet (xem `README_convert_mbpp.md`) | |
| `evalplus.Dockerfile` | Image chấm EvalPlus 0.3.1 (run_benchmark tự build khi dùng `--evaluator docker`) | |

`check_kb.py` và `clean_kb.py` thực thi code do LLM sinh (có mục gọi mạng thật, tạo hàng trăm nghìn thread): nên chạy trong Docker, VM hoặc Kaggle.
`check_contamination.py` cần `pip install evalplus scikit-learn`.
