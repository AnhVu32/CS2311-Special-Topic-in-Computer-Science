# Source_Code

Hướng dẫn đầy đủ (cài đặt, lệnh chạy, kết quả): `../README.md`. Mọi lệnh chạy từ thư mục này.

```
agent0/
├── core/          config.py      đường dẫn dữ liệu, provider, resolve_runtime(), nạp .env, luật taxonomy
│                  llm_client.py  gọi Ollama / Ollama Cloud / OpenAI-compatible, retry + backoff, seed, cache, đếm lời gọi
│                  utils.py       tách code từ markdown, parse JSON, ...
├── curriculum/    planner.py (lập subtopic, nhận phản hồi p̂), generator.py (sinh đề), strategies.py,
│                  uncertainty.py (p̂ / R_unc kiểu Agent0)
├── data/          mbpp_loader.py (đọc MBPP, chuẩn hoá test tự sinh), text_similarity.py (TF-IDF: khử trùng, khử ô nhiễm)
├── execution/     harness.py (chạy test cô lập, ghi actual/expected), sandbox.py, strict_check.py,
│                  agent.py (sinh – kiểm – chẩn đoán – sửa, đồng thuận khi nghi test sai)
├── retrieval/     retriever.py (KnowledgeRetriever: index theo hash, lọc, ngưỡng, MMR, định dạng few-shot)
├── training/      data.py (đề KB -> prompt dạng MBPP+), rewards.py (thưởng = chạy test trong harness, ADPO xấp xỉ),
│                  hf_backend.py (sinh theo lô bằng transformers) — cho RL của executor
└── scripts/       run_planner.py, run_curriculum.py, run_benchmark.py, run_demo.py
tools/             chấm hàng loạt, tổng hợp kết quả, kiểm tra / lọc KB, kiểm tra ô nhiễm (xem tools/README.md)
kaggle/            notebook Kaggle, roster.json (danh sách model), hướng dẫn (xem kaggle/README.md)
tests/             smoke_test.py (32 test, LLM / embedding / EvalPlus giả; test GRPO tự bỏ qua nếu thiếu torch/trl/peft)
```

```bash
python tests/smoke_test.py
python -m agent0.scripts.run_benchmark --help
```

## Đồng tiến hoá không huấn luyện (thay cho RL của Agent0)

Agent0 cập nhật trọng số cả hai agent bằng RL. Ở đây trọng số không đổi; thứ "tiến hoá" là KB:

| Agent0 | Ở đây |
|---|---|
| Curriculum agent học bằng GRPO, thưởng `R_unc = 1 - 2*abs(p_hat - 0.5)` | Planner nhận p̂ từng subtopic của vòng trước (`--feedback_from`) và sinh đề vào chỗ còn yếu. Đề ngoài khoảng p̂ ∈ [0.25, 0.75] bị loại; đây là lấy mẫu từ chối thay cho bước gradient |
| Executor vòng t = model đã RL ở vòng t−1 | Executor vòng t = model + KB_t: `--p_hat_kb KB_t` đo p̂ khi lời giải có kèm ví dụ truy hồi từ KB_t |
| Nhãn giả = đáp án đa số | Test được sửa bằng đồng thuận thực thi (3 lời giải độc lập cùng chống lại một test) |
| `R_rep` (phạt đề trùng) | Lọc đề gần trùng (TF-IDF) và khử ô nhiễm với MBPP+ |

Một vòng chạy như sau (vòng 1 bỏ `--feedback_from`):

```bash
python -m agent0.scripts.run_curriculum --provider ollama --model qwen2.5-coder:7b \
  --synthetic_count 300 --items_per_strategy 300 --uncertainty_samples 4 \
  --p_hat_kb ../Data/knowledge_base/<KB_t>/knowledge_base.jsonl \
  --feedback_from ../Data/knowledge_base/<vòng trước>/<model>/summary.json \
  --output_dir ../Data/knowledge_base/<vòng t+1>
```

`KB_{t+1}` là `KB_t` cộng các đề mới ở biên năng lực. Đề KB đã "phủ" thì dễ đi và bị lọc, nên biên năng lực dịch chuyển dần.

Để tách tác dụng của bước chọn theo p̂:
1. Sinh với `--p_hat_action tag`.
2. Chạy `tools/split_kb_by_phat.py` để được KB `phat` và KB `ctrl` cùng cỡ.
3. Benchmark cả hai KB (xem `kaggle/README.md`, mục 4).

## RL cho executor (GRPO + LoRA), thu nhỏ từ Agent0

Lệnh `tools/grpo_executor.py`:
- `prepare`: đề KB → prompt giống hệt prompt MBPP+ của benchmark (mô tả + 1 assert); khử ô nhiễm; chia train/val; đo p̂ của chính model rồi giữ đề có p̂ ∈ [0.25, 0.75].
- `train`: `trl.GRPOTrainer` + LoRA. Thưởng = tỉ lệ test qua trong harness + 0.1 nếu có khối code nạp được (code bị cắt cụt không được thưởng). Tuỳ chọn `--adpo`: thưởng × s(p̂) và `scale_rewards=none`, nên advantage được co giãn theo p̂ như ADPO.
- `merge`: gộp LoRA vào model gốc.
- `score`: pass@1 trên tập val.

`tools/hf_generate.py` sinh lời giải MBPP+ bằng transformers, có thể kèm LoRA và few-shot từ KB. File ra cùng định dạng với `run_benchmark`, nên chấm bằng `run_benchmark evaluate` và so bằng `summarize_matrix --model_pairs base:grpo`.

Notebook Kaggle chạy trọn quy trình: `kaggle/cs2311_grpo.ipynb` (`kaggle/README.md`, mục 8).

