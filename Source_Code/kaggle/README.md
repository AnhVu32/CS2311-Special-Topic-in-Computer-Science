# Chạy benchmark trên Kaggle

Mục tiêu: so sánh nhiều LLM — model nhỏ, ít nổi tiếng (chạy miễn phí trên GPU của Kaggle) và model lớn, nổi bật (gọi qua API) — trên MBPP+ với 4 cấu hình few-shot, để trả lời: **knowledge base (KB) giúp model nào, và giúp nhờ "truy hồi" hay chỉ nhờ "có ví dụ"?**

| File | Việc |
|---|---|
| `cs2311_kaggle.ipynb` | notebook chạy toàn bộ trên Kaggle |
| `cs2311_grpo.ipynb` | notebook thứ hai: RL cho executor kiểu Agent0 (GRPO + LoRA), rồi benchmark lại MBPP+ (mục 8) |
| `roster.json` | danh sách model (bật / tắt, thêm model) |
| `../tools/run_matrix.py` | chạy ma trận model × cấu hình, chạy tiếp được, có ngân sách thời gian |
| `../tools/summarize_matrix.py` | bảng pass@1, so sánh cặp McNemar + Holm, biểu đồ Δ theo độ mạnh của model, Δ theo độ giống của ví dụ truy hồi (kiểm tra rò rỉ), so sánh giữa các KB |
| `../tools/grpo_executor.py`, `../tools/hf_generate.py` | chuẩn bị dữ liệu / huấn luyện / chấm val cho RL; sinh lời giải MBPP+ bằng transformers |
| `../tools/split_kb_by_phat.py` | tách KB đã gắn p̂ thành KB `phat` (đề ở biên năng lực) và KB `ctrl` cùng cỡ — cho thí nghiệm Agent0 (mục 4) |
| `../tools/package_for_kaggle.py` | đóng gói project thành zip để tải lên (không kèm `.env`) |
| `dataset-metadata.json`, `kernel-metadata.json` | chỉ cần nếu dùng Kaggle CLI |

## 1. Đưa project lên Kaggle

Trên máy, từ thư mục `Project Chọn lọc`:

```bash
python3 "CS2311 Special Topic in Computer Science/Source_Code/tools/package_for_kaggle.py"
# -> Project Chọn lọc/CS2311_kaggle_upload.zip  (~1,2 MB: code, Data/mbpp, Data/knowledge_base; không có PDF, kết quả cũ, .env)
```

Kaggle → **Datasets → New Dataset** → kéo file zip vào → đặt tên `CS2311 Special Topic in Computer Science` → để **Private** → Create. Kaggle tự giải nén hay giữ nguyên zip thì notebook đều tìm được project.

Kaggle hiện gắn dataset vào `/kaggle/input/datasets/<tên tài khoản>/<slug>/` (đôi khi vẫn là `/kaggle/input/<slug>/`). Notebook tự tìm project trong `/kaggle/input` và in ra cây thư mục Input ở ô thứ hai, nên không cần sửa đường dẫn.

Sửa code xong muốn cập nhật: đóng gói lại, vào dataset → **New Version** → tải zip mới.

## 2. Tạo notebook

Kaggle → **Code → New Notebook** → **File → Import Notebook** → chọn `cs2311_kaggle.ipynb`. Ở bảng bên phải:

- **Add Input** → Your Datasets → dataset vừa tạo.
- **Session options**:
  - **Accelerator: GPU T4 x2** (hai T4, mỗi cái 16 GB). Phiên chỉ chạy model API thì để **None** cho khỏi tốn quota GPU.
  - **Internet: On** (cần xác minh số điện thoại cho tài khoản).
- **Add-ons → Secrets**: thêm key cần dùng rồi tick để gắn vào notebook. Tên key:
  - `GROQ_API_KEY`
  - `OLLAMA_API_KEY`
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`
  - `CUSTOM_API_KEY`

  Không có key nào thì notebook chỉ chạy model local.

## 3. Chạy

1. Chạy thử: ở ô cấu hình đặt `LIMIT = 10`, `ONLY = ["gemma3-4b"]` rồi **Run All** (vài phút; không chấm). Ô bắt đầu bằng `# Danh sách model sẽ chạy` in ra model nào sẽ chạy / bị bỏ (thiếu key, không có GPU) và tên model hiện có trên Groq / Ollama Cloud.
2. Chạy thật: `LIMIT = None`, `ONLY = None` → **Save Version → Save & Run All (Commit)**. Bản commit chạy nền, đóng trình duyệt vẫn chạy, tối đa 12 giờ. `TIME_BUDGET_HOURS = 10.5` để còn thời gian chấm và lưu.
   - Với **GPU T4 x2**, `LOCAL_LANES = None` tự thành 2: model local chia hai làn, hai model chạy cùng lúc, mỗi model một GPU (`run_matrix --local_lanes 2`).
3. Chưa xong trong một phiên: mở notebook → **Add Input** → tab *Notebook Output Files* → output của version trước. Đặt `PREVIOUS_RESULTS = "auto"` rồi commit lại. Notebook chép mọi thư mục `results/` cũ tìm thấy trong `/kaggle/input`; bài đã sinh được giữ, chỉ chạy phần thiếu. Có thể ghi đường dẫn cụ thể hoặc glob thay cho `"auto"`.
4. Kết quả ở tab **Output**:
   - `results/<model>/*_samples.jsonl` và kết quả chấm; `*_retrieval.jsonl` ghi ví dụ KB đã dùng cho từng bài (uid, cosine);
   - `results/summary/`: `matrix_summary.md`, `matrix_runs.csv`, `matrix_comparisons.csv`, `matrix_similarity.csv`, `matrix_delta.png`;
   - `results/logs.zip`: log từng lệnh (gom lại để output ít file);
   - `llm_cache.zip` (nếu chạy thí nghiệm Agent0): cache lời gọi LLM, phiên sau tự nạp lại;
   - `cs2311_results.zip`.

   Tải về, giải nén vào `Data/results/kaggle/` của project.

`EXCLUDE_TEST_REPAIRED = True` (mặc định) bỏ khỏi KB các mục có test từng bị sửa cho khớp lời giải — 40/431 mục của `all_models_clean`, đều do cách sửa test bằng LLM của pipeline cũ. Tên lần chạy có KB vì vậy có thêm `_notr` (vd `with_kb_n2_notr_samples.jsonl`).

Đọc `matrix_summary.md` theo thứ tự:
1. `baseline -> retrieved`: KB có giúp không.
2. `random -> retrieved`: giúp nhờ ví dụ *liên quan* hay chỉ nhờ *có ví dụ*.
3. Mục "Δ theo độ giống": lợi ích có tăng theo độ giống của ví dụ không. Nếu lợi ích chỉ dồn vào nhóm "gần trùng", đó là dấu hiệu rò rỉ dữ liệu chứ không phải tác dụng của KB.

Muốn tiết kiệm quota GPU, chia làm hai kiểu phiên:
- **Phiên GPU** chạy model local: `ONLY` = các model `provider: ollama`.
- **Phiên CPU** (Accelerator None) chạy model API. Model embedding cho cấu hình `retrieved` vẫn chạy được trên CPU.

## 4. Thí nghiệm Agent0: chọn đề ở biên năng lực có giúp không? (phiên riêng)

KB `all_models_clean` được sinh bằng pipeline cũ, chưa mục nào có p̂. Vì vậy ma trận chính chỉ trả lời "KB tự sinh có giúp không", chưa kiểm cơ chế cốt lõi của Agent0: chọn đề theo độ bất định của executor. Thí nghiệm này kiểm cơ chế đó, không cần huấn luyện:

1. `run_curriculum` với `KB_BUILDER["model"]` (mặc định `qwen2.5-coder:7b` trên GPU) sinh `synthetic_count` đề mới, kiểm chứng bằng harness và đồng thuận, rồi đo p̂ cho mọi đề. `--p_hat_action tag` giữ lại mọi đề, chỉ ghi p̂. Với `p_hat_with_kb = True`, p̂ được đo khi lời giải có kèm ví dụ truy hồi từ `KB_NAME`: executor ở đây là *model + KB hiện có*, thay cho executor đã được RL cập nhật trong Agent0.
2. `split_kb_by_phat.py` tách đề đã gắn p̂ thành:
   - KB `phat`: các đề có p̂ ∈ [0.25, 0.75];
   - KB `ctrl`: cùng số đề, chọn ngẫu nhiên trong mọi đề.

   Hai KB có cùng nguồn đề, cùng model sinh và cùng cỡ; chỉ khác ở việc có lọc theo p̂ hay không.
3. `run_matrix` chạy `baseline` và `retrieved` cho `ABLATION_MODELS` với từng KB (`--kb_tag ctrl`, `--kb_tag phat`). Baseline dùng chung, đã có thì bỏ qua.
4. `matrix_summary.md` có thêm họ so sánh `retrieved: notr_kbctrl -> notr_kbphat`. Δ > 0 có ý nghĩa nghĩa là chọn đề ở biên năng lực làm ví dụ tốt hơn.

Cách chạy: phiên GPU, đặt `RUN_MAIN_MATRIX = False`, `AGENT0_ABLATION = True`. Nếu muốn có cả so sánh với KB cũ, thêm kết quả phiên chính qua `PREVIOUS_RESULTS`.

Nếu số đề ở biên ít hơn `min_kb` (mặc định 20), notebook không chạy bước 3. Khi đó tăng `synthetic_count`. Model 7B thường giải đúng gần hết đề nó tự sinh (p̂ = 1), nên tỉ lệ đề ở biên có thể thấp.

Lặp nhiều vòng như Agent0 (chạy tay, mỗi vòng một lệnh): lấy `summary.json` và KB của vòng trước làm `--feedback_from` và `--p_hat_kb` cho vòng sau. Xem `Source_Code/README.md`, mục đồng tiến hoá không huấn luyện.

## 5. Danh sách model mặc định (`roster.json`)

| Nhóm | Model | Chạy ở đâu |
|---|---|---|
| nhỏ | qwen2.5-coder:1.5b, gemma3:4b, qwen2.5-coder:7b, llama3.1:8b, ministral-3:8b, rnj-1:8b | GPU Kaggle (Ollama) |
| vừa | gemma3:12b | GPU Kaggle (Ollama) |
| vừa | openai/gpt-oss-20b | Groq API |
| lớn | openai/gpt-oss-120b | Groq API |
| tắt sẵn | qwen/qwen3.6-27b (Groq, preview); gpt-oss:120b, qwen3.5:397b, deepseek-v4-flash (Ollama Cloud); một mẫu `custom` cho API tương thích OpenAI bất kỳ | |

- gemma3:4b, gemma3:12b, ministral-3:8b, rnj-1:8b là các model nhóm đã chạy trước đây (qua Ollama Cloud). rnj-1 là model duy nhất KB làm giảm có ý nghĩa (−3.7 điểm, p = 0.02), nên chạy lại để kiểm.
- Groq đã tắt `llama-3.1-8b-instant` và `llama-3.3-70b-versatile` cho gói Free / Developer từ 16/08/2026. `llama3.1:8b` trong danh sách là cùng trọng số nhưng chạy local.
- Ollama Cloud đã gỡ `qwen3-coder:480b`, `deepseek-v3.1`, `rnj-1:8b`... khỏi cloud (06–07/2026). Gói Free giờ chỉ có ít "starter model", nên các model cloud để tắt sẵn.
- Thêm model: một dòng trong `models`, gồm `label` (tên thư mục kết quả), `provider`, `model`, `tier`, `params_b`. Tuỳ chọn: `workers`, `max_tokens`, `extra_body`, `base_url`, `api_key_env`, `enabled`.
- Model suy luận (gpt-oss): `extra_body` `reasoning_effort: low` (Groq) hoặc `think: low` (Ollama) và `max_tokens: 4096`, để không tốn token suy nghĩ và không bị cắt mất phần code.

## 6. Thời gian và hạn mức (ước lượng — kiểm tra lại trên tài khoản)

- Kaggle: phiên GPU tối đa 12 giờ, quota GPU khoảng 30 giờ/tuần (xem ở Settings của tài khoản; tính theo thời gian phiên mở, không theo mức dùng GPU), lưu tối đa 20 GB output. Phiên CPU (Accelerator None) không tốn quota GPU.
- Image GPU của Kaggle hiện dùng Python 3.13 và `transformers>=5`; code đã chạy thử trên đúng các phiên bản này.
- Model local trên T4: mỗi lần chạy (378 bài) khoảng 20–40 phút với model 4–8B và 45–75 phút với 12B. 7 model × 4 cấu hình ≈ 15–20 giờ GPU nếu chạy lần lượt; với hai làn trên T4 x2 còn khoảng 8–10 giờ, vừa một phiên (thiếu thì phiên sau chạy tiếp).
- Thí nghiệm Agent0:
  - sinh và đo p̂ cho 300 đề với model 7B: khoảng 1–3 giờ (mỗi đề 6–8 lời gọi LLM);
  - sau đó 3 model × (baseline + 2 KB): khoảng 3–5 giờ.

  Tổng cộng vừa một phiên.
- Chấm EvalPlus: khoảng 2,5 phút mỗi lần chạy (đo trên máy 4 nhân), chạy nền song song với lúc model đang sinh.
- Groq gói Free: `openai/gpt-oss-120b` và `openai/gpt-oss-20b` mỗi model 30 request/phút, 1 000 request/ngày, 8 000 token/phút, 200 000 token/ngày (hạn mức tính riêng từng model).
  - Một lần chạy 378 bài dùng khoảng 0,2–0,5 triệu token, tức một lần chạy cần 1–3 ngày quota.
  - Nên chạy model API ở phiên CPU, mỗi ngày một lần commit với `PREVIOUS_RESULTS = "auto"`, ưu tiên `CONFIGS = ["baseline", "retrieved"]`.
  - Hết hạn mức ngày thì `run_matrix` dừng model đó ngay; chạy lại hôm sau sẽ tiếp tục.
  - Gói Developer (gắn thẻ) bỏ giới hạn ngày; một lần chạy chỉ tốn vài chục cent.
- Ollama Cloud: tính theo "usage", giới hạn theo phiên 5 giờ và theo tuần; gói Free chỉ 1 model cùng lúc.

## 7. Chạy ngoài Kaggle

Máy Linux bất kỳ có Python ≥ 3.10:
1. `pip install evalplus==0.3.1`.
2. Chạy Ollama nếu dùng model local hoặc cấu hình `retrieved`.
3. Đặt key trong `.env`.
4. Từ `Source_Code`:

```bash
python tools/run_matrix.py --roster kaggle/roster.json --results_dir ../Data/results/matrix \
  --kb_path ../Data/knowledge_base/all_models_clean/knowledge_base.jsonl --exclude_test_repaired
python tools/summarize_matrix.py --results_dir ../Data/results/matrix --roster kaggle/roster.json
```

Trên macOS, chạy `run_matrix.py --eval skip` rồi chấm từng file bằng `run_benchmark evaluate --evaluator docker`. Lý do: chấm trực tiếp trên Mac bị timeout.

## 8. RL cho executor: GRPO + LoRA (`cs2311_grpo.ipynb`)

Đây là phần "executor học bằng RL" của Agent0, thu nhỏ để vừa một GPU T4. Notebook làm các bước sau:
1. Viết lại mỗi đề trong KB đúng dạng prompt MBPP+ của benchmark và bỏ đề giống bài MBPP+.
2. Đo p̂ của chính model (4 mẫu/đề) và giữ đề ở biên năng lực, p̂ ∈ [0.25, 0.75].
3. Huấn luyện GRPO + LoRA.
4. Benchmark lại MBPP+ cho model gốc và model sau RL, mỗi model với `baseline` và `retrieved`.

| | Agent0 | Notebook |
|---|---|---|
| Executor | Qwen3-4B/8B, toàn bộ tham số | Qwen2.5-Coder-1.5B-Instruct + LoRA r=16, fp16 |
| Rollout | 16 / đề × 128 đề / bước × 40 bước × 3 vòng | 8 / đề × 4 đề / bước × ≤ 200 bước × 1 vòng |
| Thưởng | 1[đáp án = đáp án đa số] | tỉ lệ test qua (chạy trong harness của project) + 0.1 nếu có khối code nạp được (code bị cắt cụt không được thưởng) |
| Lọc đề | `abs(p̂ − 0.5) <= 0.25`, k = 10 | p̂ ∈ [0.25, 0.75], k = 4; nới dần nếu còn dưới 32 đề |
| ADPO | có | tuỳ chọn `ADPO_LIKE`: thưởng × s(p̂), không chuẩn hoá theo độ lệch chuẩn, clip trên 0.28 |
| KL | 0.01 | `KL_BETA = 0.0` mặc định (đặt 0.01 để giống paper, chậm hơn) |

**Cách chạy:**
- Notebook mới → Import `cs2311_grpo.ipynb`.
- Add Input: dataset project. Thêm output notebook chính nếu muốn dùng KB của thí nghiệm Agent0 qua `EXTRA_KBS`.
- **GPU T4 x2**, **Internet On**.
- Chạy thử trước với `LIMIT = 20`, `MAX_STEPS = 5`, rồi commit chạy thật.

Với hai GPU, GPU 1 benchmark model gốc trong lúc GPU 0 lọc đề và huấn luyện.

**Thời gian** (ước lượng, T4):

| Bước | Thời gian |
|---|---|
| Tải model | 1–2 phút |
| Lọc đề (khoảng 350 đề × 4 mẫu) | 15–20 phút |
| Huấn luyện | khoảng 1–1,5 phút/bước; dừng ở `MAX_STEPS` hoặc `TRAIN_HOURS` |
| Mỗi lần sinh 378 bài | 10–20 phút |
| Chấm EvalPlus | khoảng 3 phút/lần |

Chạy tiếp phiên trước (checkpoint lưu mỗi 25 bước): Add Input → output của version trước, `RESUME_FROM = "auto"`, và tăng `MAX_STEPS` nếu phiên trước đã chạy đủ bước. Notebook lấy lại checkpoint, tập train/val đã lọc (bỏ qua bước lọc đề) và kết quả benchmark model gốc; model sau RL được benchmark lại.

`EXTRA_KBS` nhận đường dẫn hoặc glob, vd `["/kaggle/input/**/kb_split/phat/knowledge_base.jsonl"]` để thêm KB `phat` của thí nghiệm Agent0.

**Kết quả** (`cs2311_grpo_results.zip`):
- `results/<label>-hf/` và `results/<label>-grpo/`: samples, kết quả chấm, log truy hồi.
- `results/summary/matrix_summary.md`:
  - bảng 2×2;
  - họ so sánh `model: <label>-hf -> <label>-grpo (baseline)` và `(retrieved, ...)`: McNemar giữa model gốc và model sau RL.
- `grpo/val_*.json`: pass@1 trên đề KB không dùng để huấn luyện (trong phân phối).
- `grpo/train_log.jsonl`, `train_summary.json`, biểu đồ `grpo_training.png`.
- `grpo/adapter`: LoRA, khoảng 75 MB (r = 16, fp32).

Đọc bảng 2×2 như sau:
- Hàng `-grpo` hơn `-hf` ở cột `baseline` nghĩa là RL có tác dụng.
- Cột `retrieved` hơn `baseline` nghĩa là KB có tác dụng.
- Nếu ô (grpo, retrieved) không hơn ô tốt hơn trong hai ô kia, hai cách dùng cùng một curriculum đang trùng nhau, không cộng dồn.

**Lưu ý:**
- T4 không có bf16 và FlashAttention-2, nên model chạy fp16 với attention `sdpa`. Không dùng Gemma 3 cho RL vì không ổn định ở fp16.
- `USE_VLLM = True` cài thêm vLLM để sinh rollout nhanh hơn. Trên T4, vLLM phải chạy fp16, và đã có báo cáo lỗi khi dùng LoRA. Lỗi thì để `False`; transformers vẫn chạy được, chỉ chậm hơn.
- Code đã được chạy thử trọn trên CPU với model nhỏ (smoke test và mô phỏng notebook), nhưng chưa chạy trên GPU thật. Lần đầu nên chạy thử để xem tốc độ và bộ nhớ.
- Nếu hết bộ nhớ GPU, giảm `PER_DEVICE_BATCH` (4 → 2; số bước tích luỹ gradient tự tăng, mỗi bước vẫn đủ đề), rồi `MAX_COMPLETION` (512 → 384).

## Kaggle CLI (không bắt buộc)

```bash
pip install kaggle   # cần ~/.kaggle/kaggle.json (Settings -> Create New Token)
mkdir -p /tmp/cs2311_ds && cp "Project Chọn lọc/CS2311_kaggle_upload.zip" /tmp/cs2311_ds/ \
  && cp "Project Chọn lọc/CS2311 Special Topic in Computer Science/Source_Code/kaggle/dataset-metadata.json" /tmp/cs2311_ds/
kaggle datasets create -p /tmp/cs2311_ds          # lần sau: kaggle datasets version -p /tmp/cs2311_ds -m "cập nhật"
cd "Project Chọn lọc/CS2311 Special Topic in Computer Science/Source_Code/kaggle" && kaggle kernels push -p .
```

Sửa `id` trong hai file metadata nếu tên tài khoản Kaggle khác.
