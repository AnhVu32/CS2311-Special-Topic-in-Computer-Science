# CS2311 Special Topic in Computer Science — Agent0 cho sinh code: Self-Evolving Code Knowledge Base

Hệ thống dùng LLM tự sinh đề bài lập trình, lời giải và test; kiểm chứng bằng cách chạy thử rồi lưu vào **knowledge base (KB)**. Khi giải bài mới, hệ thống truy hồi các bài tương tự trong KB làm ví dụ **few-shot (RAG)**. Đánh giá trên **MBPP / MBPP+** bằng EvalPlus.

Ý tưởng lấy từ Agent0 (bản toán, chỉ văn bản): một tác tử ra đề theo chương trình học (curriculum), một tác tử giải có công cụ chạy code. Khác Agent0, dự án **không huấn luyện mô hình** — KB là "bộ nhớ" dùng lúc suy luận.

## Cấu trúc thư mục

```
CS2311 Special Topic in Computer Science/
├── README.md                 (file này)
├── requirements.txt          thư viện cần cài
├── .env.example              mẫu file .env (API key)
├── Report/                   báo cáo (PDF)
├── Slide/                    slide (PDF)
├── Source_Code/
│   ├── agent0/               package chính
│   │   ├── core/             config (đường dẫn, provider, nạp .env), llm_client (retry/backoff), utils, logger
│   │   ├── curriculum/       planner (lập subtopic), generator (sinh đề), strategies, uncertainty (p̂ kiểu Agent0)
│   │   ├── data/             đọc MBPP, chuẩn hoá đề tự sinh, text_similarity (khử trùng / khử ô nhiễm MBPP+)
│   │   ├── execution/        agent (sinh–kiểm–sửa code, đồng thuận khi test sai), harness (chạy test cô lập), sandbox
│   │   ├── retrieval/        KnowledgeRetriever (embedding, index theo hash nội dung, ngưỡng, MMR, định dạng few-shot)
│   │   └── scripts/          run_planner, run_curriculum, run_benchmark, run_demo
│   ├── tools/                chấm hàng loạt, tổng hợp kết quả, kiểm tra / lọc KB, kiểm tra ô nhiễm dữ liệu, Dockerfile EvalPlus,
│   │                         run_matrix / summarize_matrix (nhiều model × nhiều cấu hình), package_for_kaggle
│   ├── kaggle/               notebook Kaggle + danh sách model (roster.json) + hướng dẫn
│   └── tests/smoke_test.py   kiểm tra nhanh, không cần mạng / API key
└── Data/
    ├── mbpp/                 dữ liệu gốc MBPP (+ danh sách 378 task_id của MBPP+)
    ├── knowledge_base/       KB do curriculum sinh ra, mỗi model một thư mục; all_models/ là KB gộp;
    │                         all_models_clean/ là all_models đã kiểm lại bằng harness mới (431/485 mục)
    └── results/
        ├── benchmark/        kết quả benchmark theo model (_invalid/: các lần chạy hỏng, giữ để đối chiếu)
        └── rescored_linux/   bảng chấm lại toàn bộ bằng EvalPlus trên Linux (15/11/2026)
```

Chi tiết từng thư mục dữ liệu: `Data/README.md`. Công cụ phụ: `Source_Code/tools/README.md`.

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env              # điền OLLAMA_API_KEY / GROQ_API_KEY / OPENAI_API_KEY cần dùng
ollama pull mxbai-embed-large     # model embedding cho KB, chạy trên Ollama local
```

**Mọi lệnh bên dưới chạy từ thư mục `Source_Code`:**

```bash
cd Source_Code
python tests/smoke_test.py        # 32 test, không gọi mạng, không cần API key
```

### Provider

| `--provider` | Dùng khi | Key trong `.env` | base_url mặc định |
|---|---|---|---|
| `auto` (mặc định) | tự chọn theo model và key có sẵn | — | — |
| `ollama-cloud` | `gemma3:4b`, `gemma3:12b`, `ministral-3:8b`, `qwen3-coder:480b`... | `OLLAMA_API_KEY` | `https://ollama.com` |
| `ollama` | model chạy local (`qwen2.5-coder:7b`, `llama3.1:8b`...) | — | `http://localhost:11434` |
| `groq` | `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.6-27b` (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant` đã bị Groq tắt cho gói Free / Developer từ 16/08/2026) | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| `openai` | API OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `custom` | endpoint tương thích OpenAI khác | (tuỳ) | bắt buộc `--base_url` |

`auto` chọn: có `--base_url` → `custom`; model của Groq và có `GROQ_API_KEY` → `groq`; tên dạng `gemma3:12b` → `ollama-cloud` nếu có `OLLAMA_API_KEY`, ngược lại Ollama local. Thiếu key thì báo lỗi ngay, nói rõ thiếu biến nào. File `.env` được nạp tự động (tìm ở `Source_Code/`, thư mục gốc, thư mục hiện tại); biến môi trường đã đặt sẵn được ưu tiên.

Lỗi tạm thời khi gọi LLM (429, 5xx, timeout, mất kết nối) được thử lại với exponential backoff, tôn trọng `Retry-After`; mặc định 6 lần (`AGENT0_LLM_MAX_RETRIES`). Hết hạn mức ngày / tuần / credit (vd Groq "tokens per day") thì báo lỗi ngay, không chờ; một lần sinh dừng sau `--max_consecutive_errors` bài lỗi liên tiếp (mặc định 10) và chạy tiếp được bằng `--resume`.

Tuỳ chọn cho mọi script gọi LLM: `--api_key_env TEN_BIEN` (đọc key từ biến môi trường, không để lộ key trong lệnh), `--extra_body '{...}'` (gộp vào body của request, vd `'{"reasoning_effort": "low"}'` cho gpt-oss trên Groq, `'{"options": {"num_ctx": 8192}}'` cho Ollama; giá trị `null` xoá khoá).

---

## 1. Tạo knowledge base

### Lập kế hoạch subtopic (xem trước)

```bash
python -m agent0.scripts.run_planner --domain coding --total_tasks 30 --reflection_rounds 2 \
  --provider ollama-cloud --model gemma3:4b --output ../Data/knowledge_base/plan_gemma3_4b.json
```

### Sinh KB

```bash
# Mặc định: LLM tự sinh đề (planner -> generator), giải, kiểm chứng, ghi KB
python -m agent0.scripts.run_curriculum \
  --provider ollama-cloud --model gemma3:4b \
  --synthetic_count 30 --synthetic_domain coding \
  --strategy all --items_per_strategy 10 --seed 100 \
  --log_file ../Data/knowledge_base/logs/kb_100.log

# Kiểu Agent0: thêm p̂ (k = 4 lời giải lấy mẫu / đề), chỉ giữ đề có 0.25 <= p̂ <= 0.75
python -m agent0.scripts.run_curriculum --provider ollama-cloud --model gemma3:4b \
  --synthetic_count 30 --seed 101 --uncertainty_samples 4

# Vòng sau: gửi p̂ theo subtopic của lần trước cho planner (thay cho bước cập nhật RL của Agent0)
python -m agent0.scripts.run_curriculum --provider ollama-cloud --model gemma3:4b \
  --synthetic_count 30 --seed 102 --uncertainty_samples 4 \
  --feedback_from ../Data/knowledge_base/gemma3_4b/summary.json

# Đồng tiến hoá không huấn luyện: p̂ đo trên "model + KB hiện có" (executor vòng t = model + KB_t)
python -m agent0.scripts.run_curriculum --provider ollama-cloud --model gemma3:4b \
  --synthetic_count 30 --items_per_strategy 30 --seed 103 --uncertainty_samples 4 \
  --p_hat_kb ../Data/knowledge_base/all_models_clean/knowledge_base.jsonl
```

`--synthetic_count` là số đề sinh ra. Số đề được giải còn bị chặn bởi `--items_per_strategy`: mặc định 10 đề cho mỗi chiến lược, nên với `--strategy all` thường chỉ khoảng 20–30 đề. Muốn giải hết, đặt `--items_per_strategy` không nhỏ hơn `--synthetic_count`; script sẽ cảnh báo nếu đề bị bỏ phí. Bảng ánh xạ Agent0 → bản không huấn luyện và cách lặp nhiều vòng: `Source_Code/README.md`.

Pipeline cho mỗi đề:

```
bỏ qua nếu: đã có trong KB | trùng đề khác trong lần chạy | giống một bài MBPP+ (cosine TF-IDF >= 0.5) | gần trùng mục KB (>= 0.9)
sinh code -> harness chạy test -> FAIL?
  ├─ chẩn đoán (có actual / expected của từng assert) -> sửa code (tối đa --repair_rounds lần)
  └─ vẫn FAIL -> đồng thuận (--test_repair consensus, mặc định):
       sinh --consensus_solutions lời giải độc lập (không thấy code hiện tại, không thấy assert đang tranh chấp)
       ├─ một lời giải qua hết test gốc                       -> code cũ sai, dùng lời giải đó
       ├─ mọi lời giải cho cùng một giá trị khác "expected"   -> test sai, sửa expected theo giá trị đồng thuận
       └─ không đồng thuận                                    -> loại đề
PASS -> [--uncertainty_samples k] p̂ = số lời giải lấy mẫu qua test / k; ngoài [0.25, 0.75] -> rejected.jsonl
     -> reasoning -> ghi KB
```

Harness (`agent0/execution/harness.py`) chạy lời giải trong tiến trình con `python -I`, env sạch (không thấy API key), giới hạn CPU/RAM, chặn mạng ở mức Python; nạp lời giải như module (khối `if __name__ == "__main__":` không chạy, `sys.exit` = trượt), chạy từng dòng test, đổi so sánh trần thành assert, đòi test phải gọi hàm của lời giải. Đề tự sinh cần ≥ 3 assert.

Cách cũ vẫn chọn được để so sánh: `--test_repair llm` (LLM phân xử rồi viết lại assert cho khớp code — dễ hợp thức hoá code sai), `--test_repair none` (không bao giờ sửa test).

Kết quả ghi vào `Data/knowledge_base/<model>/`: `knowledge_base.jsonl`, `rejected.jsonl` (có `reject_stage`: verification / p_hat / decontamination / exception), `summary.json` (lần chạy mới nhất: số đề nhận/loại/bỏ qua, kết quả đồng thuận, phân bố p̂, `uncertainty_by_subtopic`, số lời gọi LLM) và `summary_history.jsonl` (mọi lần chạy). `all_models/knowledge_base.jsonl` là KB gộp các model (trường `source_model`). Mỗi mục mới có `uid` theo hash nội dung, `pipeline_version: 2`, `tests_modified` (+ test gốc nếu bị sửa), `consensus`, `solution_source`, `p_hat`.

Đề MBPP làm seed: `--source mbpp` (đọc `Data/mbpp/mbpp.jsonl`, **tự bỏ 378 bài MBPP+** và các bài giống MBPP+, ghi vào `Data/knowledge_base/mbpp_seeded/<model>/`). `--provider offline --source mbpp` dựng KB từ lời giải mẫu của MBPP (không gọi LLM, ghi dưới tên `reference`) — dùng làm mốc so sánh hoặc chạy thử pipeline. `Data/knowledge_base/mbpp_seeded/knowledge_base.jsonl` là KB cũ sinh khi chưa có bộ lọc này (18 bài trùng MBPP+) — không dùng để benchmark.

Chi phí: `--cache_dir ~/.cache/agent0-llm` lưu phản hồi LLM (chạy lại cùng lệnh không tốn lời gọi); `--workers 4` giải nhiều đề song song; `explanation` dựng từ template (thêm `--llm_explanation` nếu muốn LLM viết, +1 lời gọi/đề); `--skip_reasoning` bỏ trường reasoning (-1 lời gọi/đề). Gặp 5 đề lỗi liên tiếp (vd hết quota) thì dừng và vẫn ghi summary.

### Kiểm lại KB cũ

```bash
python tools/clean_kb.py ../Data/knowledge_base/all_models/knowledge_base.jsonl \
  --out ../Data/knowledge_base/all_models_clean/knowledge_base.jsonl
```

Chạy lại từng mục bằng harness mới, bỏ mục trùng đề và mục giống MBPP+, ghi KB đã lọc + `knowledge_base.report.json`. Thêm `--drop_test_repaired` để bỏ cả các mục mà test đã bị sửa cho khớp lời giải.

---

## 2. Benchmark MBPP+ (378 bài)

Quy trình: **generate → check → evaluate → compare**.

```bash
# Sinh lời giải. Lỗi API không bao giờ thành lời giải giả; bài lỗi ghi vào *_errors.jsonl.
python -m agent0.scripts.run_benchmark generate --provider ollama-cloud --model gemma3:12b --mode baseline
python -m agent0.scripts.run_benchmark generate --provider ollama-cloud --model gemma3:12b --mode with_kb --n_examples 2

# Còn bài lỗi? Chạy lại đúng lệnh đó, thêm --resume (chỉ sinh các bài còn thiếu)
python -m agent0.scripts.run_benchmark generate --provider ollama-cloud --model gemma3:12b --mode baseline --resume

# Kiểm tra đủ 378 bài, không stub, không trùng
python -m agent0.scripts.run_benchmark check --samples_file ../Data/results/benchmark/gemma3_12b/baseline_samples.jsonl

# Chấm (trên Mac tự dùng Docker nếu có; file chưa đủ bài sẽ bị từ chối)
python -m agent0.scripts.run_benchmark evaluate --samples_file ../Data/results/benchmark/gemma3_12b/baseline_samples.jsonl

# So sánh cặp: số bài lật hai chiều + McNemar exact
python -m agent0.scripts.run_benchmark compare \
  --a ../Data/results/benchmark/gemma3_12b/baseline_samples.jsonl --a_label baseline \
  --b ../Data/results/benchmark/gemma3_12b/with_kb_n2_samples.jsonl --b_label "KB n=2"

# Hoặc làm tất cả trong một lệnh: baseline + with_kb + chấm + so sánh -> benchmark_summary.json
python -m agent0.scripts.run_benchmark run --provider ollama-cloud --model gemma3:12b --n_examples 2
```

Mặc định dùng KB `Data/knowledge_base/all_models/knowledge_base.jsonl` (đổi bằng `--kb_path`, vd `../Data/knowledge_base/all_models_clean/knowledge_base.jsonl`). Tên file tự đặt theo chế độ và n (`baseline_samples.jsonl`, `with_kb_n2_samples.jsonl`); thêm `--run_tag seed1` để có `baseline_seed1_samples.jsonl`.

`--resume` chỉ nối tiếp khi file cũ được sinh với cùng cấu hình (model, n, prompt_style, temperature, seed, cách chọn ví dụ, KB); file không có `*_meta.json` (sinh bằng bản cũ) bị từ chối — tránh trộn hai giao thức trong một file. `--workers 4` gọi model song song; `--cache_dir` cache phản hồi (temperature 0 + seed nên chạy lại là tái lập được và không tốn lời gọi).

### Few-shot từ KB và ablation

Mục KB sinh từ chính các bài đang đánh giá (theo `source_task_id` hoặc trùng đề) luôn bị loại khỏi tập truy hồi. Câu truy vấn là mô tả bài (bỏ dòng assert).

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--example_selection` | `retrieved` | `retrieved` (giống nhất theo embedding), `random` (ngẫu nhiên mỗi bài, tái lập theo seed), `fixed` (một bộ cho mọi bài) |
| `--fewshot_format` | `tests` | `tests`: đề + 1 assert + lời giải, cùng định dạng prompt MBPP+; `full`: đề + reasoning + lời giải (định dạng cũ); `code` |
| `--min_similarity` | `0` | ví dụ có cosine < τ bị bỏ; bài không còn ví dụ chạy zero-shot |
| `--mmr_lambda` | `1` | < 1 (vd 0.7) để các ví dụ không na ná nhau |
| `--exclude_test_repaired` | tắt | bỏ các mục KB mà test đã bị sửa |

Tên lần chạy tự thêm hậu tố cho lựa chọn khác mặc định (`with_kb_n2_random`, `with_kb_n2_full`, `with_kb_n2_tau0.6`...). Ví dụ đã dùng cho từng bài ghi ở `<run>_samples_retrieval.jsonl`. Bộ ablation tối thiểu để tách tác dụng "truy hồi" khỏi tác dụng "có ví dụ":

```bash
for sel in retrieved random fixed; do
  python -m agent0.scripts.run_benchmark generate --provider ollama-cloud --model gemma3:12b \
    --mode with_kb --n_examples 2 --example_selection $sel
done
```

### Chấm ở đâu

- **Linux / Kaggle:** `--evaluator local` (cần `pip install evalplus==0.3.1`). Notebook CPU của Kaggle là đủ.
- **macOS:** `--evaluator docker` (mặc định khi máy có Docker). Lần đầu tự build image `agent0-evalplus:0.3.1` từ `Source_Code/tools/evalplus.Dockerfile` (python:3.11-slim, chạy native trên chip Apple).
- **Không** chấm trực tiếp trên macOS: các lần chấm trước trên Mac bị timeout hàng loạt, MBPP+ thấp hơn thực tế 17–20 điểm.

Kết quả đọc từ `*_eval_results.json` của EvalPlus: base pass khi `base_status == "pass"`, plus pass khi cả `base_status` và `plus_status` đều là `"pass"`. pass@1 luôn tính trên **đủ 378 bài**.

### Giao thức thí nghiệm

- `--prompt_style evalplus` (mặc định): chỉ đưa prompt gốc của bài (đã có 1 assert ví dụ), đúng hướng dẫn chat của EvalPlus, so được với leaderboard. `--prompt_style with_tests` là giao thức cũ của dự án (đưa cả base test vào prompt) — chỉ dùng để tái lập số cũ và phải ghi rõ trong báo cáo.
- `--temperature 0 --seed 0` là mặc định. Chạy mỗi cấu hình ≥ 3 lần (`--seed 1 --run_tag seed1`, `--seed 2 --run_tag seed2`...) rồi báo mean ± sd.
- 1 bài = 100/378 ≈ 0.26 điểm %. Dùng `compare` để biết chênh lệch có ý nghĩa thống kê hay không:

```
b = số bài baseline fail -> KB pass ;  c = số bài baseline pass -> KB fail
p = 2 * P(X <= min(b, c)),  X ~ Binomial(b + c, 0.5)      (McNemar exact, hai phía)
```

- **Chọn số ví dụ n trên tập dev, không chọn trên MBPP+.** Tập dev là 596 bài MBPP không có trong MBPP+, chấm bằng 3 assert gốc của MBPP trong harness chặt (`agent0/execution/strict_check.py`):

```bash
python -m agent0.scripts.run_benchmark dev --provider ollama-cloud --model gemma3:12b --n_values 0 1 2 3 4 5 --dev_size 200
```

  Sau khi chọn n trên dev, chạy MBPP+ đúng một lần với n đó và báo cáo.

### File sinh ra

```
Data/results/benchmark/<model>/
├── <run>_samples.jsonl              lời giải (đầu vào của EvalPlus)
├── <run>_samples_errors.jsonl       bài lỗi API (nếu có) -> chạy lại với --resume
├── <run>_samples_retrieval.jsonl    ví dụ few-shot đã dùng cho từng bài (uid, độ giống)
├── <run>_samples_meta.json          cấu hình (cả cách chọn ví dụ), lệnh, thời gian, môi trường, hash mã nguồn, hash KB, số lời gọi LLM
├── <run>_samples_eval_results.json  kết quả từng bài của EvalPlus
├── <run>_samples_metrics.json       pass@1 base / plus trên 378 bài
└── benchmark_summary_n<n>[...].json (lệnh run) gộp baseline + with_kb + so sánh cặp
```

Repo tải về dạng zip nên không có git hash; `meta.json` ghi `code_sha256` (hash toàn bộ `agent0/`) để biết kết quả ứng với phiên bản code nào.

---

## 3. Chạy trên Kaggle: nhiều LLM, so sánh model mạnh / yếu

`Source_Code/kaggle/` có notebook `cs2311_kaggle.ipynb` chạy benchmark MBPP+ cho nhiều model trong `roster.json`:
- model nhỏ, ít nổi tiếng chạy miễn phí trên GPU T4 của Kaggle bằng Ollama: qwen2.5-coder 1.5B / 7B, gemma3 4B / 12B, llama3.1 8B, ministral-3 8B, rnj-1 8B;
- model lớn, nổi bật gọi qua API: gpt-oss-120b, gpt-oss-20b trên Groq; tuỳ chọn Ollama Cloud hoặc API tương thích OpenAI.

Mỗi model chạy 4 cấu hình: `baseline`, `retrieved`, `random`, `fixed`. KB mặc định bỏ 40 mục có test từng bị sửa cho khớp code (`--exclude_test_repaired`). Kết quả gồm:
- bảng pass@1;
- so sánh cặp McNemar + Holm;
- biểu đồ Δ theo độ mạnh của model (KB giúp model mạnh hay yếu nhiều hơn);
- bảng Δ theo độ giống của ví dụ truy hồi, để kiểm tra độ liên quan và rò rỉ.

Tuỳ chọn thêm:
- thí nghiệm Agent0 so KB chọn đề theo p̂ với KB đối chứng cùng cỡ (`kaggle/README.md`, mục 4);
- notebook thứ hai `cs2311_grpo.ipynb` (`kaggle/README.md`, mục 8): RL cho executor kiểu Agent0. Dùng GRPO + LoRA trên Qwen2.5-Coder-1.5B; thưởng chấm bằng harness của project; sau đó benchmark lại MBPP+. Kết quả là bảng 2×2: RL (tham số) × KB (ngữ cảnh).

```bash
python3 Source_Code/tools/package_for_kaggle.py     # -> ../CS2311_kaggle_upload.zip để tải lên Kaggle Datasets
```

Hướng dẫn đầy đủ (Secrets, GPU, chạy nền 12 giờ, chạy tiếp ở phiên sau, hạn mức API): `Source_Code/kaggle/README.md`. Ngoài Kaggle, cùng lệnh chạy được trên máy Linux bất kỳ (`tools/run_matrix.py`, `tools/summarize_matrix.py`).

## 4. Demo web (Gradio)

```bash
python -m agent0.scripts.run_demo --provider ollama-cloud --model gemma3:12b
python -m agent0.scripts.run_demo --provider ollama --model qwen2.5-coder:7b     # model local
```

Mở http://localhost:7860. Nhập đề tiếng Việt hoặc tiếng Anh; trả về code Python, giải thích và các bài tương tự lấy từ KB. Mặc định chỉ mở trên máy mình (`127.0.0.1`); thêm `--host 0.0.0.0` nếu muốn máy khác trong mạng truy cập. Không dựng được index KB (chưa có `mxbai-embed-large`) thì demo vẫn chạy, chỉ không dùng KB.

## 5. Dùng trong code Python

```python
from agent0.core.config import resolve_runtime
from agent0.retrieval.retriever import KnowledgeRetriever
from agent0.execution.agent import solve_with_knowledge

runtime = resolve_runtime("auto", "gemma3:12b")          # lấy key từ .env
retriever = KnowledgeRetriever(kb_path="../Data/knowledge_base/all_models/knowledge_base.jsonl")
retriever.build_index()                                 # lần đầu embed, sau đó đọc cache

task = {
    "task_id": "new_1",
    "prompt": "Write a function to find min cost path ...",
    "test_list": ["assert min_cost([[1, 2], [3, 4]], 1, 1) == 5"],
    "challenge_test_list": [],
    "test_setup_code": "",
}
few_shot = retriever.format_few_shot(retriever.query(task["prompt"], n=2, min_similarity=0.5), style="tests")
result = solve_with_knowledge(runtime, task, few_shot, repair_rounds=2)
print(result.accepted, result.solution_code)
```

---

## Kết quả

Chấm lại toàn bộ 35 file samples bằng **EvalPlus 0.3.1, MBPP+ v0.2.0, trên Linux** (15/11/2026), pass@1 trên đủ 378 bài, định dạng `base / plus`. Δ và p là so sánh cặp với baseline của cùng model (McNemar exact).

| Model | Baseline | Các biến thể KB (plus) | KB tốt nhất: Δ plus (p) |
|---|---|---|---|
| gemma3:12b | 88.4 / 74.6 | 73.8 – 75.1 | n=2: +0.5 (p = 0.85) |
| ministral-3:8b | 81.2 / 68.5 | 67.5 – 70.4 | n=1: +1.6 (p = 0.43) |
| gemma3:4b | 80.4 / 68.5 | 66.9 – 68.8 | n=2: +0.3 (p = 1.0) |
| gpt-oss:20b | 94.2 / 78.6 | 77.8 | −0.8 (p = 0.69) |
| rnj-1:8b | 89.7 / 77.2 | 73.5 | −3.7 (p = 0.02, giảm) |
| llama-3.1-8b-instant | 73.5 / 63.0 | — | — |

- Không biến thể KB nào tăng có ý nghĩa thống kê; mỗi biến thể làm lật 23–46 bài theo cả hai chiều gần bằng nhau. Kết quả duy nhất có p < 0.05 là rnj-1:8b **giảm**.
- Các số trong README trước đây (vd gemma3:12b 87.8 / 54.2) được chấm trên macOS (timeout làm MBPP+ thấp đi 17–20 điểm) và chỉ tính trên các bài không phải stub; đã được thay bằng bảng trên. Bảng đầy đủ: `Data/results/rescored_linux/`.

### Knowledge base `all_models` (485 mục)

| Model sinh | Số mục |
|---|---|
| gemma3:4b | 382 |
| qwen3-coder:480b | 58 |
| llama-3.1-8b-instant | 24 |
| llama-3.3-70b-versatile | 21 |

Kiểm tra lại bằng harness chặt (`tools/check_kb.py`): 414/485 mục (85%) thực sự được kiểm chứng; 34 mục không có phép kiểm tra nào, 8 mục có phép so sánh sai, 6 mục chỉ qua nhờ khối demo `__main__`, 6 mục gọi mạng / tạo rất nhiều thread / timeout.

`tools/clean_kb.py` (harness mới của pipeline, 16/11/2026) giữ **431/485** mục → `Data/knowledge_base/all_models_clean/`: bỏ 19 mục không có assert, 18 mục có assert sai, 5 mục mà test không gọi hàm của lời giải, 4 mục trùng đề, 8 mục giống một bài MBPP+ (diện tích hình chữ nhật ~ Mbpp/458, số ước ~ Mbpp/96, dãy con đối xứng dài nhất ~ Mbpp/247, mảng con tổng lớn nhất ~ Mbpp/559...).

---

## Tham số chính

### `run_benchmark`

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--provider`, `--model` | `auto`, `openai/gpt-oss-120b` | model đánh giá |
| `--mode` (generate) | `baseline` | `baseline` hoặc `with_kb` |
| `--n_examples` | `3` | số ví dụ few-shot lấy từ KB |
| `--kb_path` | `Data/knowledge_base/all_models/knowledge_base.jsonl` | KB dùng để truy hồi |
| `--example_selection`, `--fewshot_format` | `retrieved`, `tests` | xem mục Few-shot ở trên |
| `--min_similarity`, `--mmr_lambda` | `0`, `1` | ngưỡng độ giống, MMR |
| `--prompt_style` | `evalplus` | `evalplus` (chuẩn) hoặc `with_tests` (giao thức cũ) |
| `--temperature`, `--seed` | `0`, `0` | |
| `--resume` / `--overwrite` | — | chạy tiếp bài thiếu (cùng cấu hình) / ghi đè |
| `--run_tag` | — | hậu tố tên lần chạy |
| `--limit` | — | chỉ N bài đầu (thử nghiệm, kết quả partial) |
| `--workers`, `--cache_dir` | `1`, — | gọi model song song, cache phản hồi |
| `--evaluator` | `auto` | `local`, `docker` (auto = Docker trên Mac) |
| `--max_retries` | `6` | số lần thử lại khi gọi LLM lỗi tạm thời |

### `run_curriculum`

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--provider`, `--model` | `auto`, `openai/gpt-oss-120b` | |
| `--source` | `synthetic` | `synthetic` (LLM tự sinh đề) hoặc `mbpp` (đề MBPP, đã bỏ MBPP+); `--synthetic_only` vẫn nhận |
| `--synthetic_count`, `--synthetic_domain` | `60`, `coding` | số đề cần sinh, chủ đề cho planner |
| `--strategy`, `--items_per_strategy` | `all`, `10` | `easy_medium_hard`, `diversity`, `mutate`, `all` (mỗi đề chỉ chạy một lần) |
| `--repair_rounds`, `--verify_timeout` | `2`, `20` | số vòng sửa code, giây cho mỗi lần chạy test |
| `--test_repair`, `--consensus_solutions` | `consensus`, `2` | cách xử lý khi nghi test sai |
| `--uncertainty_samples` | `0` (tắt) | k lời giải lấy mẫu để tính p̂ (nên 4–8) |
| `--p_hat_band`, `--p_hat_action` | `0.25 0.75`, `filter` | khoảng p̂ được giữ; `tag` = chỉ ghi p̂, không loại |
| `--p_hat_kb`, `--p_hat_n_examples` | —, `2` | đo p̂ khi lời giải kèm ví dụ truy hồi từ KB này (model + KB_t) |
| `--feedback_from` | — | summary.json của lần trước → p̂ theo subtopic gửi cho planner |
| `--decontam_threshold`, `--near_dup_threshold` | `0.5`, `0.9` | bỏ đề giống MBPP+ / gần trùng KB (0 = tắt) |
| `--include_mbpp_plus` | tắt | (nguồn mbpp) cho phép cả bài MBPP+ — chỉ để thử nghiệm |
| `--skip_previously_rejected` | tắt | bỏ qua đề đã bị loại ở lần chạy trước |
| `--workers`, `--cache_dir` | `1`, — | giải song song, cache phản hồi LLM |
| `--llm_explanation`, `--skip_reasoning` | tắt, tắt | thêm / bớt lời gọi LLM cho trường explanation / reasoning |
| `--seed` | `42` | đổi seed để sinh đề khác (seed được gửi tới API) |
| `--output_dir` | `Data/knowledge_base` | KB ghi vào `<output_dir>/<model>/` |

---

## Ghi chú

- **Bản script cũ** (`curriculum_planner.py`, `executor.py`, `knowledge_retriever.py`, `benchmark_mbpp.py`, `run_agent0_mbpp_curriculum.py`, `demo.py`, `requirements_agent0_lite.txt`) nằm trong repo CS2230 (`Source_code/Agent0/Agent0_new/`). Bản này đã tách thành package `agent0/`:

| Bản cũ (CS2230) | Bản này |
|---|---|
| `curriculum_planner.py` | `agent0/curriculum/planner.py` + `scripts/run_planner.py` |
| `run_agent0_mbpp_curriculum.py` | `agent0/curriculum/*` + `scripts/run_curriculum.py` |
| `executor.py` | `agent0/execution/agent.py`, `sandbox.py`, `core/llm_client.py` |
| `knowledge_retriever.py` | `agent0/retrieval/retriever.py` |
| `benchmark_mbpp.py` | `scripts/run_benchmark.py` |
| `demo.py` | `scripts/run_demo.py` |
| `requirements_agent0_lite.txt` | `requirements.txt` |

  Tính năng của bản cũ bị rơi khi tách package đã được đưa lại: nạp `.env`, chọn provider tự động (`resolve_runtime`), `--n_examples`, `--log_level`, `--results_dir`. `--synthetic_generator` vẫn nhận để lệnh cũ chạy được, nhưng chế độ `template` không còn.
- **Hạn chế còn lại:**
  - Chặn mạng của harness là ở mức Python (chặn `socket`), không phải cách ly hệ điều hành. Để chắc chắn hơn, chạy pipeline trong Docker `--network none`.
  - Khử ô nhiễm dùng TF-IDF nên diễn đạt lại dài vẫn có thể lọt (~0.49). Vẫn nên chạy thêm `tools/check_contamination.py` trên KB trước khi benchmark.
  - p̂ tốn thêm k lời gọi cho mỗi đề được nhận.
  - Phần chính không huấn luyện: phản hồi đi vào prompt của planner, và vào KB qua p̂ đo trên model + KB (`--p_hat_kb`). RL của executor (GRPO + LoRA, `tools/grpo_executor.py`) là bản thu nhỏ so với Agent0:
    - model 1.5B, một vòng;
    - curriculum agent không được huấn luyện;
    - mới chạy thử trên CPU với model nhỏ, chưa chạy trên GPU thật.
  - Chưa chạy thử với API thật và đường chấm Docker trên máy này; mọi phần khác được kiểm bằng `tests/smoke_test.py` với LLM / embedding giả.
